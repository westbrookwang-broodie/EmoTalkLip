import torch
import platform
import math
import numpy as np
import os, cv2, argparse, subprocess

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from argparse import Namespace
from torch.utils.data import DataLoader
from python_speech_features import logfbank
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
from utils.data_avhubert import collater_audio, emb_roi2im

from models.talklip import TalkLip, TalkLip_disc_qual, DISCEMO, EmotionEncoder, Decoder


def build_encoder(hubert_root, path='config.yaml'):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)

    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertSeq2SeqConfig

    # cfg = merge_with_parent(AVHubertSeq2SeqConfig(), cfg)
    arg_overrides = {
        "dropout": cfg.dropout,
        "activation_dropout": cfg.activation_dropout,
        "dropout_input": cfg.dropout_input,
        "attention_dropout": cfg.attention_dropout,
        "mask_length": cfg.mask_length,
        "mask_prob": cfg.mask_prob,
        "mask_selection": cfg.mask_selection,
        "mask_other": cfg.mask_other,
        "no_mask_overlap": cfg.no_mask_overlap,
        "mask_channel_length": cfg.mask_channel_length,
        "mask_channel_prob": cfg.mask_channel_prob,
        "mask_channel_selection": cfg.mask_channel_selection,
        "mask_channel_other": cfg.mask_channel_other,
        "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
        "encoder_layerdrop": cfg.layerdrop,
        "feature_grad_mult": cfg.feature_grad_mult,
    }
    if cfg.w2v_args is None:
        state = checkpoint_utils.load_checkpoint_to_cpu(
            cfg.w2v_path, arg_overrides
        )
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        cfg.w2v_args = w2v_args
    else:
        state = None
        w2v_args = cfg.w2v_args
        if isinstance(w2v_args, Namespace):
            cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                w2v_args
            )

    w2v_args.task.data = cfg.data
    task_pretrain = tasks.setup_task(w2v_args.task)

    task_pretrain.load_state_dict(torch.load('task_state.pt'))

    encoder_ = task_pretrain.build_model(w2v_args.model)
    encoder = HubertEncoderWrapper(encoder_)
    if state is not None and not cfg.no_pretrained_weights:
        # set strict=False because we omit some modules
        del state['model']['mask_emb']
        encoder.w2v_model.load_state_dict(state["model"], strict=False)

    encoder.w2v_model.remove_pretraining_modules()
    return encoder, encoder.w2v_model.encoder_embed_dim


def parse_filelist(file_list, save_root, check):
    with open(file_list) as f:
        lines = f.readlines()

    if check:
        sample_paths = []
        for line in lines:
            line = line.strip().split()[0]
            if not os.path.exists('{}/{}.mp4'.format(save_root, line)):
                sample_paths.append(line)
    else:
        sample_paths = [line.strip().split()[0] for line in lines]

    return sample_paths


class Talklipdata(object):

    def __init__(self, args):
        self.data_root = args.video_root
        self.bbx_root = args.bbx_root
        self.audio_root = args.audio_root
        # self.emotion_root = args.emotion_root
        # self.datalists = parse_filelist(args.filelist, None, False)
        self.datalists = parse_filelist(args.filelist, None, False)
        self.stack_order_audio = 4
        self.train = True
        self.args = args
        self.crop_size = 96
        self.prob = 0.08
        self.length = 5
        self.emotion_dict = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
        self.text_dict = {'IEO': "It's eleven o'clock.", 'TIE': "That is exactly what happened.",
                          'IOM': "I'm on my way to the meeting.",
                          'IWW': "I wonder what this is about.", 'TAI': "The airplane is almost full.",
                          'MTI': 'Maybe tomorrow it will be cold.',
                          'IWL': "I would like a new alarm clock.", 'ITH': "I think I have a doctor's appointment.",
                          'DFA': "Don't forget a jacket.",
                          'ITS': "I think I've seen this before.", 'TSI': "The surface is slick.",
                          'WSI': "We'll stop in a couple of minutes."}

    def readtext(self, path):
        with open(path, "r") as f:
            trgt = f.readline()[7:]
        trgt = self.label_proc(trgt)
        return trgt

    def prepare_window(self, window):
        # T x 3 x H x W
        x = window / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def im_preprocess(self, ims):
        # T x 3 x H x W
        x = ims / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def filter_start_id(self, idlist):
        idlist = sorted(idlist)
        filtered = [idlist[0]]
        for item in idlist:
            if item - filtered[-1] > 4:
                filtered.append(item)
        return filtered

    def croppatch(self, images, bbxs):
        patch = np.zeros((images.shape[0], 96, 96, 3), dtype=np.float32)
        width = images.shape[1]
        for i, bbx in enumerate(bbxs):
            bbx[2] = min(bbx[2], width)
            bbx[3] = min(bbx[3], width)
            patch[i] = cv2.resize(images[i, bbx[1]:bbx[3], bbx[0]:bbx[2], :], (self.crop_size, self.crop_size))
        return patch

    def audio_visual_align(self, audio_feats, video_feats):
        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = torch.cat(
                [audio_feats, torch.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        return audio_feats

    def fre_audio(self, wav_data, sample_rate):
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames, 4 frames of tf forms a new frame of tf
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
            return feats

        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
        audio_feats = stacker(audio_feats, self.stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]
        return audio_feats

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        imgs = []
        while True:
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs

    def __len__(self):
        return len(self.datalists)

    def __getitem__(self, idx):
        """

        Args:
            idx: index of a sample in dataset

        Returns:
            inpImg: N*6*96*96
            gtImg: N*3*96*96
            spectrogram: T*104
            trgt: L
            volume: 1, which indicates T
            pickedimg: N
            imgs: T*160*160*3
            bbxs: T*4


                    sample = self.samples[idx]

        video_path = '{}/{}.mp4'.format(self.data_root, sample)
        bbx_path = '{}/{}.npy'.format(self.bbx_root, sample)
        wav_path = '{}/{}.wav'.format(self.audio_root, sample)

        bbxs = np.load(bbx_path)

        imgs = np.array(self.load_video(video_path))
        volume = len(imgs)

        sampRate, wav = wavfile.read(wav_path)
        spectrogram = self.fre_audio(wav, sampRate)
        spectrogram = torch.tensor(spectrogram) # T'* F
        with torch.no_grad():
            spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

        pickedimg = list(range(volume))
        poseImgRaw = np.array(pickedimg)
        poseImg = self.croppatch(imgs[poseImgRaw], bbxs[poseImgRaw])
        idImgRaw = np.zeros(volume, dtype=np.int32)
        idImg = self.croppatch(imgs[idImgRaw], bbxs[idImgRaw])

        poseImg = torch.tensor(poseImg, dtype=torch.float32)  # T*3*96*96
        idImg = torch.tensor(idImg, dtype=torch.float32)  # T*3*96*96

        spectrogram = self.audio_visual_align(spectrogram, imgs)

        pose_inp = self.prepare_window(poseImg)
        gt = pose_inp.clone()
        # mask off the bottom half
        pose_inp[:, :, pose_inp.shape[2] // 2:] = 0.

        id_inp = self.prepare_window(idImg)
        inp = torch.cat([pose_inp, id_inp], dim=1)

        pickedimg, bbxs = torch.tensor(pickedimg), torch.tensor(bbxs)

        imgs = torch.from_numpy(imgs)
        """
        sample = self.datalists[idx]
        print(sample)

        # S_list = sample.split("_")
        # emotion_text = S_list[2]
        # words_text = S_list[1]

        # sample = self.samples[idx]

        video_path = '{}/{}.mp4'.format(self.data_root, sample)
        bbx_path = '{}/{}.npy'.format(self.bbx_root, sample)
        words_path = '{}/{}.txt'.format(self.data_root, sample)
        wav_path = '{}/{}.wav'.format(self.audio_root, sample)
        # video_path = '{}.mp4'.format(sample)
        # bbx_path = '{}.npy'.format(sample)
        # wav_path = '{}.wav'.format(sample)

        bbxs = np.load(bbx_path)

        imgs = np.array(self.load_video(video_path))
        volume = len(imgs)

        sampRate, wav = wavfile.read(wav_path)
        spectrogram = self.fre_audio(wav, sampRate)
        spectrogram = torch.tensor(spectrogram)  # T'* F
        with torch.no_grad():
            spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

        pickedimg = list(range(volume))
        poseImgRaw = np.array(pickedimg)
        poseImg = self.croppatch(imgs[poseImgRaw], bbxs[poseImgRaw])
        idImgRaw = np.zeros(volume, dtype=np.int32)
        idImg = self.croppatch(imgs[idImgRaw], bbxs[idImgRaw])

        poseImg = torch.tensor(poseImg, dtype=torch.float32)  # T*3*96*96
        idImg = torch.tensor(idImg, dtype=torch.float32)  # T*3*96*96

        spectrogram = self.audio_visual_align(spectrogram, imgs)

        pose_inp = self.prepare_window(poseImg)
        gt = pose_inp.clone()
        # mask off the bottom half
        pose_inp[:, :, pose_inp.shape[2] // 2:] = 0.

        id_inp = self.prepare_window(idImg)
        inp = torch.cat([pose_inp, id_inp], dim=1)

        pickedimg, bbxs = torch.tensor(pickedimg), torch.tensor(bbxs)

        imgs = torch.from_numpy(imgs)

        # emotion = self.to_categorical(self.emotion_dict[emotion_text], num_classes=6)
        emotion = torch.zeros(6)

        # return inp, spectrogram, gt, volume, pickedimg, imgs, bbxs, torch.from_numpy(emotion), sample
        # label = sample.split('/')[4]
        # sample = sample.split('/')[6]

        return inp, spectrogram, gt, volume, pickedimg, imgs, bbxs, emotion, sample

    def getEmotion(self, sample):
        pass

    def to_categorical(self, y, num_classes=None, dtype='float32'):

        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y)
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def getText(self, sample):
        pass


def collate_fn(dataBatch):
    """
    Args:
        dataBatch:

    Returns:
        inpBatch: input T_sum*6*96*96, concatenation of all video chips in the time dimension
        gtBatch: output T_sum*3*96*96
        inputLenBatch: bs
        audioBatch: bs*104*T'
        audio_idx: T_sum
        targetBatch: words for lip-reading expert
        padding_mask: bs*T'
        pickedimg: a list of bs elements, each contain some picked indices
        videoBatch: a list of bs elements, each cotain a video
        bbxs: a list of bs elements
    """

    xBatch = torch.cat([data[0] for data in dataBatch], dim=0)
    yBatch = torch.cat([data[2] for data in dataBatch], dim=0)
    inputLenBatch = [data[3] for data in dataBatch]

    audioBatch, padding_mask = collater_audio([data[1] for data in dataBatch], max(inputLenBatch))

    audiolen = audioBatch.shape[2]
    idAudio = torch.cat([data[4] + audiolen * i for i, data in enumerate(dataBatch)], dim=0)

    pickedimg = [data[4] for data in dataBatch]
    videoBatch = [data[5] for data in dataBatch]
    bbxs = [data[6] for data in dataBatch]
    emotion = torch.stack([data[7] for data in dataBatch])
    sample_len = torch.tensor([len(data[0]) for data in dataBatch])
    # emotion = torch.repeat_interleave(emotion, sample_len, dim=0)
    names = [data[8] for data in dataBatch]
    # labels = [data[9] for data in dataBatch]
    # inp, spectrogram, gt, volume, pickedimg, imgs, torch.from_numpy(
    #             bbxs), torch.from_numpy(emotion), sample

    return xBatch, audioBatch, idAudio, yBatch, padding_mask, pickedimg, videoBatch, bbxs, emotion, sample_len, names


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def save_sample_images(g, gt, global_step):
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    for i, img in enumerate(g):
        cv2.imwrite('{}/{}_{}.jpg'.format(args.folder_pic_path, global_step, i), img)
    for i, img in enumerate(gt):
        cv2.imwrite('{}/{}_{}.jpg'.format(args.folder_pic_gt_path, global_step, i), img)


def model_synt(test_data_loader, device, model, args):
    tmpvideo = '{}.avi'.format(args.save_root.split('/')[-1])
    model['gen'].eval()
    model['emoGen'].eval()
    model['decoder'].eval()

    step = 0

    for inps, spectrogram, idAudio, gt, padding_mask, pickedimg, imgs, bbxs, emotion, sample_len, names in tqdm(
            test_data_loader):  #

        inps, gt = inps.to(device), gt.to(device)
        spectrogram = spectrogram.to(device)
        emotion = emotion.to(device)
        print('------emotion' + str(emotion))
        padding_mask = padding_mask.to(device)

        net_input = {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask,
                     'prev_output_tokens': None}
        sample = {'net_input': net_input, 'target_lengths': None, 'ntokens': None, 'target': None,
                  'emotion': emotion}

        x1, enc_audio = model['gen'](sample, idAudio)
        feats1, e1 = model['emoGen'](emotion, inps)
        repeated_tensor = torch.repeat_interleave(e1, torch.tensor(sample_len).to(device), dim=0).to(device)
        prediction = model['decoder'](x1, feats1, repeated_tensor, len(inps.size()), spectrogram.shape[0])
        # repeated_tensor = torch.repeat_interleave(e1, torch.tensor(sample_len_1).to(device), dim=0).to(device)

        _, height, width, _ = imgs[0].shape
        processed_img = emb_roi2im(pickedimg, imgs, bbxs, prediction, device)
        # save_sample_images(prediction, gt, global_step=step)
        # processed_img_save = (processed_img.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        # cv2.imwrite('{}/{}.jpg'.format(args.save_ic_folder, batch_idx), processed_img_save)
        # save_sample_images(inpImg1, x1e1, gtImg1, global_step,
        #                    args.checkpoint_dir)

        step += 1

        for i, video in enumerate(processed_img):
            print('NAME ' + names[i])
            out_path = '{}/{}.mp4'.format(args.save_root, names[i])
            print(out_path)

            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            out = cv2.VideoWriter(tmpvideo, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

            for j, im in enumerate(video):
                im = im.cpu().clone().detach().numpy().astype(np.uint8)
                # cv2.imwrite('{}/{}/{}.jpg'.format(args.save_pic_folder, names[i], j), im)
                out.write(im)

            out.release()

            audio = '{}/{}.wav'.format(args.audio_root, names[i])

            # audio = '/home/star/lipread_mp4/{}/test/{}.wav'.format(labels[i], names[i])

            command = '{} -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel quiet'.format(args.ffmpeg, audio, tmpvideo,
                                                                                      out_path)

            subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Synthesize videos to be evaluated')

    parser.add_argument('--filelist', help="Path of a file list containing all samples' name", required=True, type=str)
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument("--audio_root", help="Root folder of audio", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--save_root", help="a directory to save synthesized videos", required=True, type=str)
    parser.add_argument('--ckpt_path', help='pretrained checkpoint', required=True, type=str)
    parser.add_argument('--emo_ckpt_path', help='pretrained emoGen checkpoint', required=True, type=str)
    parser.add_argument('--decoder_ckpt_path', help='pretrained decoder checkpoint', required=True, type=str)
    parser.add_argument('--avhubert_root', help='Path of av_hubert root', required=True, type=str)
    parser.add_argument('--check', help='whether filter out videos which have been synthesized in save_root',
                        default=True, type=bool)
    parser.add_argument('--ffmpeg', default='ffmpeg', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--folder_pic_path', default='/home/star/wy/G_new', type=str)
    parser.add_argument('--folder_pic_gt_path', default='/home/star/wy/GT_new', type=str)

    args = parser.parse_args()

    device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

    # Dataset and Dataloader setup
    test_dataset = Talklipdata(args)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, num_workers=6)  # hparams.batch_size, 4,

    gen = TalkLip(*build_encoder(args.avhubert_root), 768).to(device)
    emoGen = EmotionEncoder().to(device)
    decoder = Decoder().to(device)
    gen.load_state_dict(torch.load(args.ckpt_path, map_location=device)["state_dict"])
    emoGen.load_state_dict(torch.load(args.emo_ckpt_path, map_location=device)["state_dict"])
    decoder.load_state_dict(torch.load(args.decoder_ckpt_path, map_location=device)["state_dict"])

    model = {
        'gen': gen,
        'emoGen': emoGen,
        'decoder': decoder
    }
    with torch.no_grad():
        model_synt(test_loader, device, model, args)
