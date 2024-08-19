import torch
import platform
import math
import numpy as np
import os, cv2, argparse, subprocess
from os.path import dirname, join, basename
from torch import optim
from models.talklip import TalkLip, TalkLip_disc_qual, DISCEMO, EmotionEncoder, Decoder

from tqdm import tqdm
from torch.nn import functional as F
from argparse import Namespace
from python_speech_features import logfbank
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
from utils.data_avhubert import collater_audio, emb_roi2im
from backgroundremover.bg import remove
from PIL import Image
import io

from models.talklip import TalkLip
import face_detection
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import sys

sys.path.append(os.path.split(sys.path[0])[0])
for pth in sys.path:
    print(pth)
from DiffBIR.model.cldm import ControlLDM
from DiffBIR.model.gaussian_diffusion import Diffusion
from DiffBIR.model.bsrnet import RRDBNet
from DiffBIR.model.scunet import SCUNet
from DiffBIR.model.swinir import SwinIR
from DiffBIR.utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from DiffBIR.utils.face_restoration_helper import FaceRestoreHelper
from DiffBIR.utils.helpers import (
    Pipeline,
    BSRNetPipeline, SwinIRPipeline, SCUNetPipeline,
    bicubic_resize
)
from DiffBIR.utils.cond_fn import MSEGuidance, WeightedMSEGuidance

from accelerate.utils import set_seed
from DiffBIR.utils.inference import (
    V1InferenceLoop,
    BSRInferenceLoop, BFRInferenceLoop, BIDInferenceLoop, UnAlignedBFRInferenceLoop
)

emotion_dict = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}


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


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def prepare_window(window):
    # T x 3 x H x W
    x = window / 255.
    x = x.permute((0, 3, 1, 2))

    return x


def detect_bbx(frames, fa):
    height, width, _ = frames[0].shape
    batches = [frames[i:i + 32] for i in range(0, len(frames), 32)]

    bbxs = list()
    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            if f is None:
                htmp = int((height - 96) / 2)
                wtmp = int((width - 96) / 2)
                x1, y1, x2, y2 = wtmp, htmp, wtmp + 96, htmp + 96
            else:
                x1, y1, x2, y2 = f
            bbxs.append([x1, y1, x2, y2])
    bbxs = np.array(bbxs)
    return bbxs


def croppatch(images, bbxs, crop_size=96):
    patch = np.zeros((images.shape[0], crop_size, crop_size, 3))
    width = images.shape[1]
    for i, bbx in enumerate(bbxs):
        bbx[2] = min(bbx[2], width)
        bbx[3] = min(bbx[3], width)
        patch[i] = cv2.resize(images[i, bbx[1]:bbx[3], bbx[0]:bbx[2], :], (crop_size, crop_size))
    return patch


def audio_visual_pad(audio_feats, video_feats):
    diff = len(audio_feats) - len(video_feats)
    repeat = 1
    if diff > 0:
        repeat = math.ceil(len(audio_feats) / len(video_feats))
        video_feats = torch.repeat_interleave(video_feats, repeat, dim=0)

    diff = len(audio_feats) - len(video_feats)
    if diff == 0:
        diff = len(video_feats)
    video_feats = video_feats[:diff]
    return video_feats, repeat, diff


def fre_audio(wav_data, sample_rate):
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

    from python_speech_features import logfbank
    if len(wav_data.shape) > 1:
        audio_feats = logfbank(wav_data[:, 0], samplerate=sample_rate).astype(np.float32)  # [T, F]
    else:
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, 4)  # [T/stack_order_audio, F*stack_order_audio]
    return audio_feats


def load_video(path):
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


def data_preprocess(args, face_detector):
    video_path = args.video_path
    wav_path = args.wav_path

    imgs = np.array(load_video(video_path))

    sampRate, wav = wavfile.read(wav_path)
    spectrogram = fre_audio(wav, sampRate)
    spectrogram = torch.tensor(spectrogram)  # T'* F
    with torch.no_grad():
        spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

    bbxs = detect_bbx(imgs, face_detector)

    poseImg = croppatch(imgs, bbxs)

    poseImg = torch.tensor(poseImg, dtype=torch.float32)  # T*3*96*96

    poseImg, repeat, diff = audio_visual_pad(spectrogram, poseImg)

    if repeat > 1:
        imgs = np.repeat(imgs, repeat, axis=0)
        bbxs = np.repeat(bbxs, repeat, axis=0)
    imgs = imgs[:diff]
    bbxs = bbxs[:diff]

    pose_inp = prepare_window(poseImg)
    id_inp = pose_inp.clone()
    # mask off the bottom half
    # pose_inp[:, :, pose_inp.shape[2] // 2:] = 0.
    pose_inp[:, :, :] = 0.

    inp = torch.cat([pose_inp, id_inp], dim=1)

    bbxs = torch.tensor(bbxs)
    imgs = torch.from_numpy(imgs)

    audioBatch = spectrogram.unsqueeze(dim=0).transpose(1, 2)
    padding_mask = (torch.BoolTensor(1, inp.shape[0]).fill_(False))
    idAudio = torch.arange(inp.shape[0])
    if args.emotion is None:
        emotion = torch.zeros(6)
    else:
        emotion = to_categorical(emotion_dict[args.emotion], num_classes=6)
    emotion = torch.tensor(emotion)
    emotion = emotion.repeat(inp.shape[0], 1)
    print('------emotion')
    print(emotion)

    return inp, audioBatch, idAudio, padding_mask, [imgs], [bbxs], emotion


def to_categorical(y, num_classes=None, dtype='float32'):
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


def synt_demo(face_detector, device, model, emotion_encoder, decoder, args):
    model.eval()

    inps, spectrogram, idAudio, padding_mask, imgs, bbxs, emotion = data_preprocess(args, face_detector)

    # print('imgs',imgs[1].)

    inps = inps.to(device)
    spectrogram = spectrogram.to(device)
    padding_mask = padding_mask.to(device)
    emotion = emotion.to(device)
    print(emotion.shape)

    sample = {'net_input': {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask,
                            'prev_output_tokens': None},
              'target_lengths': None, 'ntokens': None, 'target': None, 'emotion': emotion, 'sample_len': None}

    pred_1, _ = model(sample, idAudio)
    pred_2, e = emotion_encoder(emotion, inps)
    prediction = decoder(pred_1, pred_2, e, len(inps.size()), spectrogram.shape[0])
    emo_out = emoDisc(prediction)
    print(emo_out)

    # _, height, width, _ = imgs[0].shape
    _, height, width, _ = 96,96,96,96
    save_sample_images(inps, prediction, 1000, args.save_dir)
    # processed_img = emb_roi2im([idAudio], bbxs, prediction.cpu(), 'cpu')
    print()



    out_path = '{}.mp4'.format(args.save_path)
    tmpvideo = '{}.avi'.format(args.save_path)

    out = cv2.VideoWriter(tmpvideo, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
    print('iiiiiiiiiiiiiiiiiiiiiiiiiiiii')

    list_input = []
    prediction = (prediction.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    for j, im in enumerate(prediction):
        # im = im.cpu().clone().detach().numpy().astype(np.uint8)
        # im = im.astype(np.uint8)
        list_input.append(im)
        # args.input = im
        # im = BFRInferenceLoop(args).run()
        # # 将NumPy数组转换为PIL图像对象
        # im_pil = Image.fromarray(im)
        #
        # # 将PIL图像转换为字节流
        # img_byte_arr = io.BytesIO()
        # im_pil.save(img_byte_arr, format='PNG')  # 将图像保存为PNG格式的字节流
        # img_byte_arr = img_byte_arr.getvalue()  # 获取字节数据
        #
        # # 调用remove函数，去除背景
        # result = remove(img_byte_arr, model_name='u2net',
        #                 alpha_matting=True,
        #                 alpha_matting_foreground_threshold=240,
        #                 alpha_matting_background_threshold=10,
        #                 alpha_matting_erode_structure_size=10,
        #                 alpha_matting_base_size=1000)
        #
        # f = open('remove_{}.jpg'.format(j), "wb")
        # f.write(result)
        # f.close()

        # # 将处理后的字节流转换回PIL图像
        # result_img = Image.open(io.BytesIO(result))

        # 将PIL图像转换回NumPy数组，确保与OpenCV兼容
        # result_np = np.array(result_img)
        # print(result_np)
        # # 移除Alpha通道
        # if result_np.shape[2] == 4:  # 检查是否有Alpha通道
        #     result_np = cv2.cvtColor(result_np, cv2.COLOR_RGBA2RGB)
        # result_np = cv2.resize(result_np, (width, height))
        #
        # # # 可选：对图像进行高斯模糊处理
        # # result_np = cv2.GaussianBlur(result_np, (5, 5), 0)
        #
        # # 使用OpenCV写入文件或视频流
        # out.write(result_np)
        # im = im.encode()
        # im = remove(im, model_name='u2net',
        #              alpha_matting=True,
        #              alpha_matting_foreground_threshold=240,
        #              alpha_matting_background_threshold=10,
        #              alpha_matting_erode_structure_size=10,
        #              alpha_matting_base_size=1000)

        # im = cv2.GaussianBlur(im, (5, 5), 0)
        # out.write(im)
    args.input = list_input
    sample_outs = BFRInferenceLoop(args).run()
    for sample_out in sample_outs:
        out.write(sample_out)

    out.release()

    command = '{} -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel quiet'.format(args.ffmpeg, args.wav_path, tmpvideo,
                                                                              out_path)  #

    subprocess.call(command, shell=platform.system() != 'Windows')


def save_sample_images(x, g, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g), axis=-2)
    for batch_idx, c in enumerate(collage):
        cv2.imwrite('{}/{}.jpg'.format(folder, batch_idx), c)


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = model.state_dict()
    for k, v in s.items():
        new_s[k] = v
    model.load_state_dict(new_s)

    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
    else:
        global_step = 0


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

    return global_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Synthesize a video conditioned on a different audio')

    parser.add_argument("--video_path", help="Root folder of video", required=True, type=str)
    parser.add_argument("--wav_path", help="Root folder of audio", required=True, type=str)
    parser.add_argument("--save_path", help="a directory to save the synthesized video", default='Real_hap', type=str)
    parser.add_argument('--ckpt_path', help='pretrained checkpoint', required=True,
                        default='/ssd2/m3lab/usrs/wy/TalkLip/restart/checkpoint_step000148000.pth', type=str)
    parser.add_argument('--emotion_encoder_ckpt_path', help='pretrained checkpoint', required=True,
                        default='/ssd2/m3lab/usrs/wy/TalkLip/restart/emoGen_checkpoint_step000148000.pth', type=str)
    parser.add_argument('--decoder_ckpt_path', help='pretrained checkpoint', required=True,
                        default='/ssd2/m3lab/usrs/wy/TalkLip/restart/decoder_checkpoint_step000148000.pth', type=str)
    parser.add_argument('--avhubert_root', help='Path of av_hubert root', required=True, type=str)
    parser.add_argument('--check', help='whether filter out videos which have been synthesized in save_root',
                        default=False, type=bool)
    parser.add_argument('--ffmpeg', default='ffmpeg', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--emotion', default=None, type=str, required=False)
    parser.add_argument('--save_dir', default='/home/star/wy/imgs', type=str)
    parser.add_argument("--task", type=str, required=True, choices=["sr", "dn", "fr", "fr_bg"])
    parser.add_argument("--upscale", type=float, required=True)
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    ### sampling parameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--better_start", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str, default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    ### input parameters
    parser.add_argument("--input", type=str)
    parser.add_argument("--n_samples", type=int, default=1)
    ### guidance parameters
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="w_mse", choices=["mse", "w_mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_start", type=int, default=1001)
    parser.add_argument("--g_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=1)
    ### output parameters
    parser.add_argument("--output", type=str, required=True)
    ### common parameters
    parser.add_argument("--seed", type=int, default=231)
    # parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    # torch.cuda.empty_cache()

    device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

    model = TalkLip(*build_encoder(args.avhubert_root)).to(device)
    emotion_encoder = EmotionEncoder().to(device)
    decoder = Decoder().to(device)
    emoDisc = DISCEMO().to(device)
    emo_dis_optimizer = optim.Adam([p for p in emoDisc.parameters() if p.requires_grad], lr=1e-4, betas=(0.5, 0.999))
    load_checkpoint('/home/star/wy/emo_disc_ checkpoint/emo_disccheckpoint_step000062000.pth', emoDisc,
                    emo_dis_optimizer,
                    reset_optimizer=False, overwrite_global_states=False)
    # emoDisc.load_state_dict(torch.load('/ssd2/m3lab/usrs/wy/EmoGen/emo-checkpoint/disc_emo_52000.pth'))

    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                      device=device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device)["state_dict"])
    emotion_encoder.load_state_dict(torch.load(args.emotion_encoder_ckpt_path, map_location=device)["state_dict"])
    decoder.load_state_dict(torch.load(args.decoder_ckpt_path, map_location=device)["state_dict"])
    with torch.no_grad():
        synt_demo(fa, device, model, emotion_encoder, decoder, args)
