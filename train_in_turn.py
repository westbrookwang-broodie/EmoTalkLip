from os.path import dirname, join, basename
from tqdm import tqdm

from inf_test import parse_filelist
from models.talklip import TalkLip, TalkLip_disc_qual, DISCEMO, EmotionEncoder, Decoder

import torch
import gc
import logging
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from argparse import Namespace
from torch.utils.data import DataLoader
from python_speech_features import logfbank
from fairseq.data import data_utils
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
from utils.data_avhubert import collater_audio, images2avhubert, emb_roi2im_train, emb_roi2im
from collections import defaultdict
import os, random, cv2, argparse, subprocess
from torch.nn.utils.rnn import pad_sequence
import copy
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import albumentations as A


def init_logging(level=logging.INFO,
                 log_name='log/sys.log',
                 formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')):
    logger = logging.getLogger()
    logger.setLevel(level=level)
    handler = logging.FileHandler(log_name)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def build_encoder(hubert_root, cfg):
    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertSeq2SeqConfig

    cfg = merge_with_parent(AVHubertSeq2SeqConfig(), cfg)
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
    if state is not None:
        task_pretrain.load_state_dict(state['task_state'])
    # task_pretrain.state = task.state

    encoder_ = task_pretrain.build_model(w2v_args.model)
    encoder = HubertEncoderWrapper(encoder_)
    if state is not None and not cfg.no_pretrained_weights:
        # set strict=False because we omit some modules
        del state['model']['mask_emb']
        encoder.w2v_model.load_state_dict(state["model"], strict=False)

    encoder.w2v_model.remove_pretraining_modules()
    return encoder


def get_avhubert(hubert_root, ckptpath):
    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_pretraining import LabelEncoderS2SToken
    from fairseq.dataclass.utils import DictConfig

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckptpath])
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.report_accuracy = True

    dictionaries = [task.target_dictionary]
    bpe_tokenizer = task.s2s_tokenizer
    procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
    extra_gen_cls_kwargs = {
        "lm_model": None,
        "lm_weight": 0.0,
    }
    arg_gen = DictConfig({'_name': None, 'beam': 50, 'nbest': 1, 'max_len_a': 1.0, 'max_len_b': 0, 'min_len': 1,
                          'match_source_len': False, 'unnormalized': False, 'no_early_stop': False,
                          'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None,
                          'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0,
                          'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None,
                          'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5,
                          'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None,
                          'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10,
                          'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1,
                          'iter_decode_with_external_reranker': False, 'retain_iter_history': False,
                          'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None,
                          'no_seed_provided': False})
    generator = task.build_generator(
        models, arg_gen, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )
    encoder = build_encoder(hubert_root, saved_cfg.model)
    model_dict_avhubert = models[0].state_dict()
    model_dict_encoder = encoder.state_dict()
    for key in model_dict_encoder.keys():
        model_dict_encoder[key] = model_dict_avhubert['encoder.' + key]
    encoder.load_state_dict(model_dict_encoder)
    return models[0], procs[0], generator, criterion, encoder


def retrieve_avhubert(hubert_root, hubert_ckpt, device):
    avhubert, label_proc, generator, criterion, encoder = get_avhubert(hubert_root, hubert_ckpt)
    """Base configuration"""
    ftlayers = list(range(9, 12))

    ftlayers_full = ['w2v_model.encoder.layers.' + str(layer) for layer in ftlayers]
    for name, p in encoder.named_parameters():
        ft_ind = False
        for layer in ftlayers_full:
            if layer in name:
                ft_ind = True
                break
        if ft_ind:
            p.requires_grad = True
        else:
            p.requires_grad = False

    for p in avhubert.parameters():
        p.requires_grad = False
    avhubert = avhubert.to(device)
    avhubert.eval()
    return avhubert, label_proc, generator, criterion, encoder


def fil_train(data):
    return (data.split('_')[3] == 'HI') and int(
        data.split('_')[0]) <= 1070


def fil_test(data):
    return data.split('_')[3] == 'HI' and int(data.split('_')[0]) > 1070


def fil_ref(data):
    return data.split('/')[4].startswith('A')


class Talklipdata_2(object):

    def __init__(self, split, args, label_proc):
        # self.emotion_root = args.emotion_root
        self.label_proc = label_proc
        # self.datalists = list(
        #     filter(fil_ref, parse_filelist('{}/{}.txt'.format(args.file_dir_2, split), None, False)))
        self.datalists = parse_filelist('{}/{}.txt'.format(args.file_dir_2, split), None, False)
        self.stack_order_audio = 4
        self.train = True
        self.args = args
        self.crop_size = 96
        self.prob = 0.08
        self.length = 5
        self.emotion_dict = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
        self.text_dict = {'IEO': "It's eleven o'clock", 'TIE': "That is exactly what happened",
                          'IOM': "I'm on my way to the meeting",
                          'IWW': "I wonder what this is about", 'TAI': "The airplane is almost full",
                          'MTI': 'Maybe tomorrow it will be cold',
                          'IWL': "I would like a new alarm clock", 'ITH': "I think I have a doctor's appointment",
                          'DFA': "Don't forget a jacket",
                          'ITS': "I think I've seen this before", 'TSI': "The surface is slick",
                          'WSI': "We'll stop in a couple of minutes"}
        # target = {}
        # for i in range(1, 2 * emonet_T):
        #     target['image' + str(i)] = 'image'

        # self.augments = A.Compose([
        #     A.RandomBrightnessContrast(p=0.4),
        #     A.RandomGamma(p=0.4),
        #     A.CLAHE(p=0.4),
        #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.4),
        #     A.ChannelShuffle(p=0.4),
        #     A.RGBShift(p=0.4),
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        # ], additional_targets=target, p=0.8)
        # self.content_to_audio = defaultdict(list)
        # for sample in self.datalists:
        #     content_label = sample.split("_")[1] + '_' + sample.split("_")[0]
        #     # content_label = sample.split("_")[1]
        #     self.content_to_audio[content_label].append(sample)

        # 生成音频对
        # self.pairs = self.generate_pairs()

    def readtext(self, path):
        with open(path, "r") as f:
            trgt = f.readline()[7:]
        trgt = self.label_proc(trgt)
        return trgt

    def im_preprocess(self, ims):
        # args = {}
        # args['image'] = video[0, :, :, :]
        # for i in range(1, 2*emonet_T):
        #     args['image' + str(i)] = video[i, :, :, :]
        # result = self.augments(**args)
        # video[0, :, :, :] = result['image']
        # for i in range(1, 2*emonet_T):
        #     video[i, :, :, :] = result['image' + str(i)]
        ##T x 3 x H x W
        x = ims / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def im_preprocess_1(self, ims):
        # T x 3 x H x W
        x = ims / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def augmentVideo(self, video):
        args = {}
        args['image'] = video[0, :, :, :]
        for i in range(1, 2 * emonet_T):
            args['image' + str(i)] = video[i, :, :, :]
        result = self.augments(**args)
        video[0, :, :, :] = result['image']
        for i in range(1, 2 * emonet_T):
            video[i, :, :, :] = result['image' + str(i)]
        return video

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

    def load_video_spec(self, sample):
        cap = cv2.VideoCapture('{}.mp4'.format(sample))
        txt = '{}.txt'.format(sample)
        with open(txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find('Duration') != -1:
                    duration = float(line.split(' ')[1])

        fps = cap.get(5)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        mid_frame = frame_count // 2
        start_frame = max(mid_frame - int(duration * fps), 0)
        end_frame = min(mid_frame + int(duration * fps), frame_count)
        imgs = []
        # 跳转到开始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 读取帧并写入新的视频文件
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                imgs.append(frame)
            current_frame += 1

        # 释放所有资源
        cap.release()

        bbx_path = '{}.npy'.format(sample)
        bbx_data = np.load(bbx_path)
        middle_bbx = bbx_data[start_frame:end_frame]

        return imgs, middle_bbx, duration

    def load_middle_wav_segment(self, wav_path, n_seconds):
        # 读取整个 WAV 文件
        sampRate, wav = wavfile.read(wav_path)

        # 计算音频的总时长（以秒为单位）
        duration = len(wav) / sampRate

        # 计算音频的中间点
        middle_sample = len(wav) // 2

        # 计算截取范围的开始和结束样本数
        start_sample = max(middle_sample - int(n_seconds * sampRate), 0)
        end_sample = min(middle_sample + int(n_seconds * sampRate), len(wav))

        # 提取中间前后 n 秒的音频数据
        middle_wav_segment = wav[start_sample:end_sample]

        return sampRate, middle_wav_segment

    def __len__(self):
        return len(self.datalists)

    def __getitem__(self, idx):
        sample = self.datalists[idx]
        words_text = self.readtext('{}/{}.txt'.format(args.video_root, sample))
        # words_text = sample.split('/')[4].upper()
        wav_path = '{}/{}.wav'.format(args.audio_root, sample)
        bbx_path = '{}/{}.npy'.format(args.bbx_root, sample)
        video_path = '{}/{}.mp4'.format(args.video_root, sample)

        bbxs = np.load(bbx_path)
        imgs = self.load_video(video_path)
        # imgs, bbxs, duration = self.load_video_spec(sample)
        imgs = np.array(imgs)
        volume = len(imgs)
        # bbxs = np.load(bbx_path)
        # sampRate, wav = self.load_middle_wav_segment(wav_path, duration)
        sampRate, wav = wavfile.read(wav_path)
        spectrogram = self.fre_audio(wav, sampRate)
        spectrogram = torch.from_numpy(spectrogram)
        with torch.no_grad():
            spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

        if self.train:
            pid_start = random.sample(list(range(1, volume - 4)), max(int(volume * self.prob), 1))
        else:
            pid_start = list(range(0, volume - 4, int(volume * 0.12)))

        pid_start = self.filter_start_id(pid_start)
        pid_start = np.array(pid_start)

        poseidx, ididx = [], []

        for i, index in enumerate(pid_start):
            poseidx += list(range(index, index + self.length))
            wrongindex = random.choice(list(range(volume - 4)))
            while wrongindex == index:
                wrongindex = random.choice(list(range(volume - 4)))
            ididx += list(range(wrongindex, wrongindex + self.length))

        if not self.train:
            ididx = np.zeros(len(poseidx), dtype=np.int32)

        pickedimg = poseidx

        poseImg = self.croppatch(imgs[poseidx], bbxs[poseidx])
        idImg = self.croppatch(imgs[ididx], bbxs[ididx])

        poseImg = torch.from_numpy(poseImg)
        idImg = torch.from_numpy(idImg)

        trgt = words_text

        spectrogram = self.audio_visual_align(spectrogram, imgs)

        if dataAug is False:
            poseImg = self.im_preprocess_1(poseImg)
            y = poseImg.clone()
            poseImg[:, :, poseImg.shape[2] // 2:] = 0.
            # poseImg[:, :, :poseImg.shape[2] // 2] = 0.
            # poseImg[:, :, :] = 0.
            idImg = self.im_preprocess_1(idImg)
            x = torch.cat([poseImg, idImg], axis=1)

        if dataAug is True:
            poseImg = np.asarray(poseImg).astype(np.uint8)
            y = poseImg.copy()
            poseImg[:, poseImg.shape[1] // 2:, :] = 0.
            # poseImg[:, :, :] = 0.

            # idImg = self.im_preprocess(idImg)
            idImg = np.asarray(idImg).astype(np.uint8)
            inpImg = np.concatenate([y, idImg], axis=0)
            aug_results = self.augmentVideo(inpImg)
            y, wrong_window = np.split(aug_results, 2, axis=0)
            y = self.im_preprocess(y)
            poseImg = self.im_preprocess(poseImg)
            wrong_window = self.im_preprocess(wrong_window)

            # y = np.transpose(y, (3, 0, 1, 2)) / 255
            # window = np.transpose(window, (3, 0, 1, 2))
            # wrong_window = np.transpose(wrong_window, (3, 0, 1, 2)) / 255

            x = np.concatenate([poseImg, wrong_window], axis=0)
            x = torch.FloatTensor(x)
            x = x.permute((1, 0, 2, 3))
            y = torch.FloatTensor(y)
            y = y.permute((1, 0, 2, 3))

        pickedimg = torch.tensor(pickedimg)
        # emotion = self.to_categorical(random.randint(0, 5), num_classes=6)
        emotion = torch.zeros(6)

        # emotion = self.to_categorical(self.emotion_dict[emotion_text], num_classes=6)

        return x, spectrogram, y, trgt, volume, pickedimg, torch.from_numpy(imgs), torch.from_numpy(
            bbxs), emotion, sample

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


# emonet_T = 5
dataAug = True


class Talklipdata(object):

    def __init__(self, split, args, label_proc, train):
        self.augments = None
        self.data_root = args.video_root
        self.bbx_root = args.bbx_root
        self.audio_root = args.audio_root
        self.text_root = args.word_root
        # self.emotion_root = args.emotion_root
        self.label_proc = label_proc
        if train:
            self.datalists = list(
                filter(fil_train, parse_filelist('{}/{}.txt'.format(args.file_dir, split), None, False)))
        else:
            self.datalists = list(
                filter(fil_test, parse_filelist('{}/{}.txt'.format(args.file_dir, split), None, False)))
        # self.datalists = list(
        #     filter(fil_ref, parse_filelist('{}/{}.txt'.format(args.file_dir, split), None, False)))
        # self.datalists =  self.datalists)
        self.stack_order_audio = 4
        self.train = True
        self.args = args
        self.crop_size = 96
        self.prob = 0.08
        self.length = 5
        self.emotion_dict = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
        self.text_dict = {'IEO': "It's eleven o'clock", 'TIE': "That is exactly what happened",
                          'IOM': "I'm on my way to the meeting",
                          'IWW': "I wonder what this is about", 'TAI': "The airplane is almost full",
                          'MTI': 'Maybe tomorrow it will be cold',
                          'IWL': "I would like a new alarm clock", 'ITH': "I think I have a doctor's appointment",
                          'DFA': "Don't forget a jacket",
                          'ITS': "I think I've seen this before", 'TSI': "The surface is slick",
                          'WSI': "We'll stop in a couple of minutes"}
        # target = {}
        # for i in range(1, 2 * emonet_T):
        #     target['image' + str(i)] = 'image'

        # self.augments = A.Compose([
        #     A.RandomBrightnessContrast(p=0.4),
        #     A.RandomGamma(p=0.4),
        #     A.CLAHE(p=0.4),
        #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.4),
        #     A.ChannelShuffle(p=0.4),
        #     A.RGBShift(p=0.4),
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        # ], additional_targets=target, p=0.8)
        self.content_to_audio = defaultdict(list)
        for sample in self.datalists:
            content_label = sample.split("_")[1] + '_' + sample.split("_")[0]
            # content_label = sample.split("_")[1]
            self.content_to_audio[content_label].append(sample)

        # 生成音频对
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        pairs = []
        for content, samples in self.content_to_audio.items():
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    if samples[i].split("_")[2] != samples[j].split("_")[2]:  # 确保情绪不同
                        pairs.append((samples[i], samples[j]))
        print('Pairs: ', len(pairs))
        return pairs

    def readtext(self, path):
        with open(path, "r") as f:
            trgt = f.readline()[7:]
        trgt = self.label_proc(trgt)
        return trgt

    def im_preprocess(self, ims):
        # args = {}
        # args['image'] = video[0, :, :, :]
        # for i in range(1, 2*emonet_T):
        #     args['image' + str(i)] = video[i, :, :, :]
        # result = self.augments(**args)
        # video[0, :, :, :] = result['image']
        # for i in range(1, 2*emonet_T):
        #     video[i, :, :, :] = result['image' + str(i)]
        ##T x 3 x H x W
        x = ims / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def im_preprocess_1(self, ims):
        # T x 3 x H x W
        x = ims / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def augmentVideo(self, video, frame_cnt):
        frame_cnt = video.shape[0]
        target = {f'image{i}': 'image' for i in range(1, frame_cnt)}
        self.augments = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.CLAHE(p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.2),
            A.ChannelShuffle(p=0.2),
            A.RGBShift(p=0.2),
            A.RandomBrightness(p=0.2),
            A.RandomContrast(p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
        ], additional_targets=target, p=0.8)
        args = {}
        args['image'] = video[0, :, :, :]
        for i in range(1, frame_cnt):
            args['image' + str(i)] = video[i, :, :, :]
        result = self.augments(**args)
        video[0, :, :, :] = result['image']
        for i in range(1, frame_cnt):
            video[i, :, :, :] = result['image' + str(i)]
        return video

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
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        imgs = []
        while True:
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs, frame_count

    def load_video_spec(self, sample):
        cap = cv2.VideoCapture('{}.mp4'.format(sample))
        txt = '{}.txt'.format(sample)
        with open(txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find('Duration') != -1:
                    duration = float(line.split(' ')[1])

        fps = cap.get(5)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        mid_frame = frame_count // 2
        start_frame = max(mid_frame - int(duration * fps), 0)
        end_frame = min(mid_frame + int(duration * fps), frame_count)
        imgs = []
        # 跳转到开始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 读取帧并写入新的视频文件
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                imgs.append(frame)
            current_frame += 1

        # 释放所有资源
        cap.release()

        bbx_path = '{}.npy'.format(sample)
        bbx_data = np.load(bbx_path)
        middle_bbx = bbx_data[start_frame:end_frame]

        return imgs, middle_bbx, duration

    def load_middle_wav_segment(self, wav_path, n_seconds):
        # 读取整个 WAV 文件
        sampRate, wav = wavfile.read(wav_path)

        # 计算音频的总时长（以秒为单位）
        duration = len(wav) / sampRate

        # 计算音频的中间点
        middle_sample = len(wav) // 2

        # 计算截取范围的开始和结束样本数
        start_sample = max(middle_sample - int(n_seconds * sampRate), 0)
        end_sample = min(middle_sample + int(n_seconds * sampRate), len(wav))

        # 提取中间前后 n 秒的音频数据
        middle_wav_segment = wav[start_sample:end_sample]

        return sampRate, middle_wav_segment

    def __len__(self):
        return len(self.datalists)

    # def __getitem__(self, idx):
    #     sample = self.datalists[idx]
    #     words_text = sample.split('/')[4].upper()
    #     video_path = '{}.mp4'.format(sample)
    #     wav_path = '{}.wav'.format(sample)
    #     bbx_path = '{}.npy'.format(sample)
    #     # bbxs = np.load(bbx_path)
    #     # imgs = np.load(self.load_video(video_path))
    #     imgs, bbxs, duration = self.load_video_spec(sample)
    #     imgs = np.array(imgs)
    #     volume = len(imgs)
    #     sampRate, wav = self.load_middle_wav_segment(wav_path, duration)
    #     # sampRate, wav = wavfile.read(wav_path)
    #     spectrogram = self.fre_audio(wav, sampRate)
    #     spectrogram = torch.from_numpy(spectrogram)
    #     with torch.no_grad():
    #         spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])
    #
    #     if self.train:
    #         pid_start = random.sample(list(range(1, volume - 4)), max(int(volume * self.prob), 1))
    #     else:
    #         pid_start = list(range(0, volume - 4, int(volume * 0.12)))
    #
    #     pid_start = self.filter_start_id(pid_start)
    #     pid_start = np.array(pid_start)
    #
    #     poseidx, ididx = [], []
    #
    #     for i, index in enumerate(pid_start):
    #         poseidx += list(range(index, index + self.length))
    #         wrongindex = random.choice(list(range(volume - 4)))
    #         while wrongindex == index:
    #             wrongindex = random.choice(list(range(volume - 4)))
    #         ididx += list(range(wrongindex, wrongindex + self.length))
    #
    #     if not self.train:
    #         ididx = np.zeros(len(poseidx), dtype=np.int32)
    #
    #     pickedimg = poseidx
    #
    #     poseImg = self.croppatch(imgs[poseidx], bbxs[poseidx])
    #     idImg = self.croppatch(imgs[ididx], bbxs[ididx])
    #
    #     poseImg = torch.from_numpy(poseImg)
    #     idImg = torch.from_numpy(idImg)
    #
    #     trgt = self.label_proc(words_text)
    #
    #     spectrogram = self.audio_visual_align(spectrogram, imgs)
    #
    #     if dataAug is False:
    #         poseImg = self.im_preprocess_1(poseImg)
    #         y = poseImg.clone()
    #         poseImg[:, :, poseImg.shape[2] // 2:] = 0.
    #         # poseImg[:, :, :poseImg.shape[2] // 2] = 0.
    #         # poseImg[:, :, :] = 0.
    #         idImg = self.im_preprocess_1(idImg)
    #         x = torch.cat([poseImg, idImg], axis=1)
    #
    #     if dataAug is True:
    #         poseImg = np.asarray(poseImg).astype(np.uint8)
    #         y = poseImg.copy()
    #         poseImg[:, poseImg.shape[1] // 2:, :] = 0.
    #         # poseImg[:, :, :] = 0.
    #
    #         # idImg = self.im_preprocess(idImg)
    #         idImg = np.asarray(idImg).astype(np.uint8)
    #         inpImg = np.concatenate([y, idImg], axis=0)
    #         aug_results = self.augmentVideo(inpImg)
    #         y, wrong_window = np.split(aug_results, 2, axis=0)
    #         y = self.im_preprocess(y)
    #         poseImg = self.im_preprocess(poseImg)
    #         wrong_window = self.im_preprocess(wrong_window)
    #
    #         # y = np.transpose(y, (3, 0, 1, 2)) / 255
    #         # window = np.transpose(window, (3, 0, 1, 2))
    #         # wrong_window = np.transpose(wrong_window, (3, 0, 1, 2)) / 255
    #
    #         x = np.concatenate([poseImg, wrong_window], axis=0)
    #         x = torch.FloatTensor(x)
    #         x = x.permute((1, 0, 2, 3))
    #         y = torch.FloatTensor(y)
    #         y = y.permute((1, 0, 2, 3))
    #
    #     pickedimg = torch.tensor(pickedimg)
    #     # emotion = self.to_categorical(random.randint(0, 5), num_classes=6)
    #     emotion = torch.zeros(6)
    #
    #     # emotion = self.to_categorical(self.emotion_dict[emotion_text], num_classes=6)
    #
    #     return x, spectrogram, y, trgt, volume, pickedimg, torch.from_numpy(imgs), torch.from_numpy(
    #         bbxs), emotion, sample

    def __getitem__(self, idx):
        sample1, sample2 = self.pairs[idx]

        def process_sample(sample):
            S_list = sample.split("_")
            emotion_text = S_list[2]
            words_text = S_list[1]

            video_path = '{}/{}.mp4'.format(self.data_root, sample)
            bbx_path = '{}/{}.npy'.format(self.bbx_root, sample)
            wav_path = '{}/{}.wav'.format(self.audio_root, sample)
            bbxs = np.load(bbx_path)
            imgs, frame_cnt = self.load_video(video_path)
            imgs = np.array(imgs)  # T*96*96*3
            volume = len(imgs)
            print(volume)

            sampRate, wav = wavfile.read(wav_path)
            spectrogram = self.fre_audio(wav, sampRate)
            spectrogram = torch.from_numpy(spectrogram)  # T'* F, T'*104
            with torch.no_grad():
                spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

            # if self.train:
            #     LI = list(range(1, volume - 4))
            #     pid_start = np.random.choice(LI, max(int(volume * self.prob), 1))
            # else:
            #     pid_start = list(range(0, volume - 4, int(volume * 0.12)))
            #
            # pid_start = self.filter_start_id(pid_start)
            # pid_start = np.array(pid_start)

            if self.train:
                pid_start = random.sample(list(range(1, volume - 4)), max(int(volume * self.prob), 1))
            else:
                pid_start = list(range(0, volume - 4, int(volume * 0.12)))

            pid_start = self.filter_start_id(pid_start)
            pid_start = np.array(pid_start)

            poseidx, ididx = [], []

            for i, index in enumerate(pid_start):
                poseidx += list(range(index, index + self.length))
                wrongindex = random.choice(list(range(volume - 4)))
                while wrongindex == index:
                    wrongindex = random.choice(list(range(volume - 4)))
                ididx += list(range(wrongindex, wrongindex + self.length))

            if not self.train:
                ididx = np.zeros(len(poseidx), dtype=np.int32)

            pickedimg = poseidx

            poseImg = self.croppatch(imgs[poseidx], bbxs[poseidx])
            idImg = self.croppatch(imgs[ididx], bbxs[ididx])

            poseImg = torch.from_numpy(poseImg)
            idImg = torch.from_numpy(idImg)

            trgt = self.label_proc(self.text_dict[words_text].upper())

            spectrogram = self.audio_visual_align(spectrogram, imgs)

            if dataAug is False:
                poseImg = self.im_preprocess_1(poseImg)
                y = poseImg.clone()
                # poseImg[:, :, poseImg.shape[2] // 2:] = 0.
                # poseImg[:, :, :poseImg.shape[2] // 2] = 0.
                poseImg[:, :, :] = 0.
                idImg = self.im_preprocess_1(idImg)
                x = torch.cat([poseImg, idImg], axis=1)

            if dataAug is True:
                poseImg = np.asarray(poseImg).astype(np.uint8)
                y = poseImg.copy()
                # poseImg[:, poseImg.shape[1] // 2:, :] = 0.
                poseImg[:, :, :] = 0.

                # idImg = self.im_preprocess(idImg)
                idImg = np.asarray(idImg).astype(np.uint8)
                inpImg = np.concatenate([y, idImg], axis=0)
                aug_results = self.augmentVideo(inpImg, frame_cnt)
                y, wrong_window = np.split(aug_results, 2, axis=0)
                y = self.im_preprocess(y)
                poseImg = self.im_preprocess(poseImg)
                wrong_window = self.im_preprocess(wrong_window)

                # y = np.transpose(y, (3, 0, 1, 2)) / 255
                # window = np.transpose(window, (3, 0, 1, 2))
                # wrong_window = np.transpose(wrong_window, (3, 0, 1, 2)) / 255

                x = np.concatenate([poseImg, wrong_window], axis=0)
                x = torch.FloatTensor(x)
                x = x.permute((1, 0, 2, 3))
                y = torch.FloatTensor(y)
                y = y.permute((1, 0, 2, 3))

            pickedimg = torch.tensor(pickedimg)

            emotion = self.to_categorical(self.emotion_dict[emotion_text], num_classes=6)

            return x, spectrogram, y, trgt, volume, pickedimg, torch.from_numpy(imgs), torch.from_numpy(
                bbxs), torch.from_numpy(emotion), sample

        inpImg1, spectrogram1, gtImg1, trgt1, volume1, pickedimg1, imgs1, bbxs1, emotion1, sample1 = process_sample(
            sample1)
        inpImg2, spectrogram2, gtImg2, trgt2, volume2, pickedimg2, imgs2, bbxs2, emotion2, sample2 = process_sample(
            sample2)

        return (inpImg1, spectrogram1, gtImg1, trgt1, volume1, pickedimg1, imgs1, bbxs1, emotion1, sample1), (
            inpImg2, spectrogram2, gtImg2, trgt2, volume2, pickedimg2, imgs2, bbxs2, emotion2, sample2)

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


def collater_seq_label_s2s(targets):
    lengths = torch.LongTensor([len(t) for t in targets])
    ntokens = lengths.sum().item()
    pad, eos = 1, 2
    targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
    prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False,
                                                   move_eos_to_beginning=True)
    return (targets_, prev_output_tokens), lengths, ntokens


def collater_label(targets_by_label):
    targets_list, lengths_list, ntokens_list = [], [], []
    itr = zip(targets_by_label, [-1], [1])
    for targets, label_rate, pad in itr:
        if label_rate == -1:
            targets, lengths, ntokens = collater_seq_label_s2s(targets)
        targets_list.append(targets)
        lengths_list.append(lengths)
        ntokens_list.append(ntokens)
    return targets_list[0], lengths_list[0], ntokens_list[0]


def collater_audio_single(audio, audio_size, audio_start=None):
    audio_feat_shape = list(audio.shape[1:])
    collated_audio = audio.new_zeros([audio_size] + audio_feat_shape)
    padding_mask = (torch.BoolTensor(1, audio_size).fill_(False))
    start_known = audio_start is not None
    audio_start = 0 if not start_known else audio_start

    diff = len(audio) - audio_size
    if diff == 0:
        collated_audio = audio
    elif diff < 0:
        # assert pad_audio
        collated_audio = torch.cat([audio, audio.new_full([-diff] + audio_feat_shape, 0.0)])
        padding_mask[diff:] = True
    else:
        import sys
        sys.exit('Audio segment is longer than the loggest')

    if len(audio.shape) == 2:
        collated_audio = collated_audio.transpose(0, 1)  # [T, F] -> [F, T]
    elif len(audio.shape) == 4:
        collated_audio = collated_audio.permute((3, 0, 1, 2)).contiguous()  # [T, H, W, C] -> [C, T, H, W]
    else:
        collated_audio = collated_audio.permute((2, 0, 1)).contiguous()

    return collated_audio, padding_mask


def pad_data(batch):
    # Assuming each element in batch is a tuple (face_data, emotion_data, target_data)

    return face_data_padded, emotion_data, target_data_padded


def collate_fn(data_batch):
    """
    Args:
        data_batch: 由一组内容相同的数据对组成的批次数据，每个元素是一个元组，包含两个样本的数据对

    Returns:
        inp_batch: 输入数据的批次，T_sum*6*96*96，将两个样本的视频数据在时间维度上连接起来
        gt_batch: 输出数据的批次，T_sum*3*96*96
        audio_batch: 音频数据的批次，bs*104*T'，bs 是批次大小，T' 是音频序列的最大长度
        audio_idx: 表示音频序列在拼接后的音频批次中的索引，长度为 T_sum
        target_batch: 目标文本的批次，包含了两个样本的目标文本
        padding_mask: 表示音频序列的填充掩码，形状为 (bs, T')，bs 是批次大小，T' 是音频序列的最大长度
        picked_img: 包含了两个样本的选定图像索引，是一个列表
        video_batch: 包含了两个样本的视频数据，是一个列表
        bbxs: 包含了两个样本的边界框数据，是一个列表
        emotion: 包含了两个样本的情感数据，形状为 (bs, num_classes)，num_classes 是情感类别数
        sample_len: 包含了每个样本的长度，形状为 (bs,)
        sample_name: 包含了每个样本的名称，是一个列表
    """
    # 分别提取数据对中的第一个和第二个样本的数据
    data1 = [data[0] for data in data_batch]
    data2 = [data[1] for data in data_batch]

    inpBatch_1 = torch.cat([data[0] for data in data1], dim=0)
    inpBatch_2 = torch.cat([data[0] for data in data2], dim=0)
    gt_batch_1 = torch.cat([data[2] for data in data1], dim=0)
    gt_batch_2 = torch.cat([data[2] for data in data2], dim=0)
    inputLenBatch_1 = [data[4] for data in data1]
    inputLenBatch_2 = [data[4] for data in data2]

    audioBatch_1, padding_mask_1 = collater_audio([data[1] for data in data1], max(inputLenBatch_1))
    audioBatch_2, padding_mask_2 = collater_audio([data[1] for data in data2], max(inputLenBatch_2))
    audio_idx_1 = torch.cat([data[5] + audioBatch_1.shape[2] * i for i, data in enumerate(data1)], dim=0)
    audio_idx_2 = torch.cat([data[5] + audioBatch_2.shape[2] * i for i, data in enumerate(data2)], dim=0)

    targetBatch_1 = collater_label([[data[3] for data in data1]])
    targetBatch_2 = collater_label([[data[3] for data in data2]])

    bbxs_1 = [data[7] for data in data1]
    bbxs_2 = [data[7] for data in data2]
    pickedimg_1 = [data[5] for data in data1]
    pickedimg_2 = [data[5] for data in data2]
    videoBatch_1 = [data[6] for data in data1]
    videoBatch_2 = [data[6] for data in data2]

    emotion1 = torch.stack([data[8] for data in data1], dim=0)
    emotion2 = torch.stack([data[8] for data in data2], dim=0)

    sample_len_1 = [len(data[0]) for data in data1]
    sample_len_2 = [len(data[0]) for data in data2]
    sample_name_1 = [data[9] for data in data1]
    sample_name_2 = [data[9] for data in data2]

    # # 将情感数据拼接成列表
    # emotion = torch.tensor([data[8].detach().numpy() for data in data_batch])
    # sample_len = torch.tensor([len(data[0]) for data in data_batch])
    # emotion = torch.repeat_interleave(emotion, sample_len, dim=0)
    #
    # # 将样本名称拼接成列表
    # sample_name = list(sample_name1) + list(sample_name2)

    return inpBatch_1, audioBatch_1, audio_idx_1, gt_batch_1, targetBatch_1, padding_mask_1, pickedimg_1, videoBatch_1, bbxs_1, emotion1, sample_len_1, sample_name_1, inpBatch_2, audioBatch_2, audio_idx_2, gt_batch_2, targetBatch_2, padding_mask_2, pickedimg_2, videoBatch_2, bbxs_2, emotion2, sample_len_2, sample_name_2


# x, spectrogram, y, trgt, volume, pickedimg, torch.from_numpy(imgs), torch.from_numpy(
#             bbxs), sample
def collate_fn_re(dataBatch):
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
    inpBatch = torch.cat([data[0] for data in dataBatch], dim=0)
    gtBatch = torch.cat([data[2] for data in dataBatch], dim=0)
    inputLenBatch = [data[4] for data in dataBatch]

    audioBatch, padding_mask = collater_audio([data[1] for data in dataBatch], max(inputLenBatch))
    audio_idx = torch.cat([data[5] + audioBatch.shape[2] * i for i, data in enumerate(dataBatch)], dim=0)

    targetBatch = collater_label([[data[3] for data in dataBatch]])

    bbxs = [data[7] for data in dataBatch]
    pickedimg = [data[5] for data in dataBatch]
    videoBatch = [data[6] for data in dataBatch]

    emotions = torch.stack([data[8] for data in dataBatch], dim=0)
    sample = [data[9] for data in dataBatch]
    sample_len = [len(data[0]) for data in dataBatch]

    return inpBatch, audioBatch, audio_idx, gtBatch, targetBatch, padding_mask, pickedimg, videoBatch, bbxs, emotions, inputLenBatch, sample, sample_len

    # # emotion = torch.Tensor([data[8] for data in dataBatch])
    #
    # return inpBatch, audioBatch, audio_idx, gtBatch, targetBatch, padding_mask, pickedimg, videoBatch, bbxs, emotion, sample_len, sample_name


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        cv2.imwrite('{}/{}.jpg'.format(folder, batch_idx), c)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def local_sync_loss(pickid, enc_audio, enc_video):
    pickedAud = enc_audio.permute(1, 0, 2).reshape(-1, enc_audio.shape[2])[pickid]
    pickedVid = enc_video.permute(1, 0, 2).reshape(-1, enc_video.shape[2])[pickid]
    return pickedVid, pickedAud


def freezeNet(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreezeNet(network):
    for p in network.parameters():
        p.requires_grad = True


class status_manager(object):
    def __init__(self, patience=5, status=0):
        self.min = 100.
        self.waited_itr = 0
        self.status = status
        self.patience = patience

    def update(self, performance):
        if performance < self.min:
            self.min = performance
            self.waited_itr = 0
        else:
            self.waited_itr += 1

    def check_status(self):
        if self.waited_itr > self.patience:
            self.status += 1
            self.waited_itr = 0
            return self.status, True
        else:
            return self.status, False

    # train(device, {'gen': imGen, 'disc': imDisc}, avhubert, criterion,
    #       {'train': train_data_loader, 'test': test_data_loader},
    #       {'gen': optimizer, 'disc': disc_optimizer}, args, global_step, logger)


def dtw_distance(x1, x2):
    dtw_distances = []
    # 将张量的非时间维度展平为1维，形状变为 [time, features]
    x1_flatten = x1.view(x1.size(0), -1)
    print('x1_flattn' + str(x1_flatten.shape))
    x2_flatten = x2.view(x2.size(0), -1)
    # 遍历每个特征维度计算 DTW 距离
    for i in range(x1_flatten.size(1)):  # 遍历每个特征维度
        distance, _ = fastdtw(x1_flatten[:, i].cpu().numpy(), x2_flatten[:, i].cpu().numpy(), dist=euclidean)
        dtw_distances.append(distance)
    return sum(dtw_distances)


def emo_contrastive_loss(x1emotion1, x2emotion1, x1emotion2, x2emotion2, margin=1.0):
    # 计算同一对的欧几里得距离
    same_pair_distance_1 = torch.norm(x1emotion1 - x2emotion1, p=2)
    same_pair_distance_2 = torch.norm(x1emotion2 - x2emotion2, p=2)

    # 计算不同对的欧几里得距离
    diff_pair_distance_1 = torch.norm(x1emotion1 - x2emotion2, p=2)
    diff_pair_distance_2 = torch.norm(x2emotion1 - x1emotion2, p=2)

    # 对比损失
    loss_same = torch.mean(same_pair_distance_1 ** 2 + same_pair_distance_2 ** 2)
    loss_diff = torch.mean(
        torch.clamp(margin - diff_pair_distance_1, min=0) ** 2 + torch.clamp(margin - diff_pair_distance_2, min=0) ** 2)

    loss = loss_same + loss_diff
    return loss


def get_emotions(syntims, sample_lens, emo_disc, real_emotions, Eloss):
    loss = 0.0
    video_fragments = torch.split(syntims, sample_lens, dim=0)
    emotions = []
    for i, fragment in enumerate(video_fragments):
        emotion = emo_disc(fragment)
        emotions.append(emotion)
        loss += Eloss(emotion, torch.argmax(real_emotions[i], dim=0))
    emotions = torch.stack(emotions)
    return loss, emotions


def just_get_emotions(syntims, sample_lens, emo_disc):
    video_fragments = torch.split(syntims, sample_lens, dim=0)
    emotions = []
    for i, fragment in enumerate(video_fragments):
        emotion = emo_disc(fragment)
        emotions.append(emotion)
    emotions = torch.stack(emotions)
    max_indices = torch.argmax(emotions, dim=1)
    res = torch.nn.functional.one_hot(max_indices, num_classes=6).float()

    return res


def train_ref(device, model, avhubert, criterion, data_loader, optimizer, args, global_step, logger):
    print('Starting Step: {}'.format(global_step))

    lip_train = True
    emo_train = True
    model['gen'].ft = False
    status = status_manager(5)
    recon_loss = nn.L1Loss()
    l1_loss = nn.L1Loss()
    emo_loss_disc = nn.CrossEntropyLoss()

    accumulation_steps = 4

    # ((inpBatch_1, audioBatch_1, audio_idx_1, gt_batch_1, targetBatch_1, padding_mask_1, pickedimg_1, videoBatch_1,
    #   bbxs_1, emotion1, sample_len_1, sample_name_1),
    #  (inpBatch_2, audioBatch_2, audio_idx_2, gt_batch_2, targetBatch_2, padding_mask_2, pickedimg_2, videoBatch_2,
    #   bbxs_2, emotion2, sample_len_2, sample_name_2))
    for epoch in range(args.n_epoch):
        losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'emo_loss': 0, 'emo_cont_loss': 0, 'prec_g': 0, 'disc_real_g': 0,
                  'disc_fake_g': 0, 'emo_disc_loss': 0, 'c_l1': 0}
        prog_bar = tqdm(enumerate(data_loader['train']))
        print(len(data_loader['train']))
        oom_count = 0

        ##inpBatch, audioBatch, audio_idx, gtBatch, targetBatch, padding_mask, pickedimg, videoBatch, bbxs, emotions, inputLenBatch
        for step, (
                inpBatch, audioBatch, audio_idx, gtBatch, ((trgt, prev_trg), tlen, ntoken), padding_mask, pickedimg,
                videoBatch,
                bbxs, emotions, inputLenBatch, sample, sample_len) in prog_bar:
            for key in model.keys():
                model[key].train()
            print('SAMPLE')
            print(sample)
            # print('SAMPLE2')
            # print(sample2)
            inpBatch = inpBatch.to(device)
            audioBatch = audioBatch.to(device)
            audio_idx = audio_idx.to(device)
            gtBatch = gtBatch.to(device)
            trgt, prev_trg = trgt.to(device), prev_trg.to(device)
            padding_mask = padding_mask.to(device)
            criterion.report_accuracy = False
            emotions = emotions.to(device)
            freezeNet(model['disc'])
            freezeNet(model['emo_disc'])
            for key in optimizer.keys():
                optimizer[key].zero_grad()

            net_input = {'source': {'audio': audioBatch, 'video': None}, 'padding_mask': padding_mask,
                         'prev_output_tokens': prev_trg}
            sample = {'net_input': net_input, 'target_lengths': tlen, 'ntokens': ntoken, 'target': trgt,
                      'emotion': emotions}

            print('------------------------------------------------')

            ###    {'gen': imGen, 'emoGen': emGen, 'decoder': decoder, 'disc': imDisc, 'emo_disc': emoDisc},
            ###    {'gen': gen_optimizer, 'emo_gen': emo_optimizer, 'decoder':decoder_optimizer, 'disc': disc_optimizer, 'emo_dis': emo_dis_optimizer},

            try:
                # mock_emotions = just_get_emotions(gtBatch, sample_len, model['emo_disc'])
                mock_emotions = emotions
                print('GET_EMOTIONS')
                print(mock_emotions)
                mock_emotions = mock_emotions.to(device)
                x1, x1_embad = model['gen'](sample, audio_idx)  # g: T*3*96*96
                feats1, e1 = model['emoGen'](mock_emotions, inpBatch)
                # print('EEEE1')
                # print(e1.shape)
                # print('xxx1')
                # print(x1.shape)
                # print(inputLenBatch)
                # # 使用 torch.repeat_interleave() 函数复制每个样本的情绪嵌入对应次数
                repeated_tensor = torch.repeat_interleave(e1, torch.tensor(sample_len).to(device), dim=0).to(device)
                # expanded_tensor_repeat = e1.repeat(int(x1.shape[0]/e1.shape[0]), 1, 1, 1)

                x1e1 = model['decoder'](x1, feats1, repeated_tensor, len(inpBatch.size()), audioBatch.shape[0])

                ##train emo_disc

                ##train generator
                if lip_train:
                    processed_img = images2avhubert(pickedimg, videoBatch, bbxs, x1e1, audioBatch.shape[2], device)
                    sample['net_input']['source']['video'] = processed_img
                    sample['net_input']['source']['audio'] = None
                    lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample)
                    losses['lip'] += lip_loss.item()
                    if args.cont_w > 0:
                        pickedVid, pickedAud = local_sync_loss(audio_idx, x1_embad, enc_out['encoder_out'])
                        local_sync = model['gen'].sync_net(torch.squeeze(pickedVid), torch.squeeze(pickedAud))
                        losses['local_sync'] += local_sync.item()
                    else:
                        local_sync = 0.
                else:
                    lip_loss, local_sync = 0., 0.

                if args.perp_w > 0.:
                    perceptual_loss = model['disc'].perceptual_forward(x1e1)
                    losses['prec_g'] += perceptual_loss.item()  ##gan
                else:
                    perceptual_loss = 0.
                #
                if args.e_w_ref > 0:
                    loss1, x1emotion1 = get_emotions(x1e1, sample_len, model['emo_disc'], mock_emotions, emo_loss_disc)
                    emo_loss = loss1
                    print('X1EMOTION1' + str(x1emotion1))
                    print('E1' + str(mock_emotions))
                    emo_cont_loss = 0.0
                    #
                    losses['emo_loss'] += emo_loss.item()
                    # losses['emo_cont_loss'] += emo_cont_loss.item()
                else:
                    emo_loss = 0.0
                    emo_cont_loss = 0.0

                l1loss = recon_loss(x1e1, gtBatch)
                losses['l1'] += l1loss.item()

                ##contro_loss
                # c_l1loss = 0.0
                if args.c_l1_w > 0:
                    if x1e2.shape[0] < gtImg2.shape[0]:
                        c_l1loss = recon_loss(x1e2, gtImg2[:int(x1e2.shape[0]), :, :, :])
                    elif x1e2.shape[0] > gtImg2.shape[0]:
                        c_l1loss = recon_loss(x1e2[: int(gtImg2.shape[0]), :, :, :], gtImg2)
                    else:
                        c_l1loss = recon_loss(x1e2, gtImg2)

                    if x2e1.shape[0] < gtImg1.shape[0]:
                        c_l1loss += recon_loss(x2e1, gtImg1[:int(x2e1.shape[0]), :, :, :])
                    elif x2e1.shape[0] > gtImg1.shape[0]:
                        c_l1loss += recon_loss(x2e1[: int(gtImg1.shape[0]), :, :, :], gtImg1)
                    else:
                        c_l1loss += recon_loss(x2e1, gtImg1)
                    losses['c_l1'] += c_l1loss.item()
                else:
                    c_l1loss = 0.0
                    # c_l1loss = recon_loss(x1e2, gtImg2)
                    # c_l1loss += recon_loss(x2e1, gtImg1)

                #
                loss = args.lip_w * lip_loss + args.perp_w * perceptual_loss + (
                        1. - args.lip_w - args.perp_w - args.e_w) * l1loss + args.cont_w * local_sync + args.e_w * emo_loss
                + args.e_c * emo_cont_loss + args.c_l1_w * c_l1loss
                # loss = loss / accumulation_steps
                loss.backward()

                # if (step + 1) % accumulation_steps == 0:
                optimizer['gen'].step()
                optimizer['gen'].zero_grad()
                optimizer['emo_gen'].step()
                optimizer['emo_gen'].zero_grad()
                optimizer['decoder'].step()
                optimizer['decoder'].zero_grad()

                unfreezeNet(model['disc'])
                # # unfreezeNet(model['emo_disc'])
                # disc_loss_1, real_emotion_1 = get_emotions(gtImg1, sample_len_1, model['emo_disc'], emotion1,
                #                                            emo_loss_disc)
                # print('REAL_EMOTION1' + str(real_emotion_1))
                #
                # # emo_disc_real_loss = emo_loss_disc(pred, torch.argmax(emotion1[0]))
                # disc_loss_2, real_emotion_2 = get_emotions(gtImg2, sample_len_2, model['emo_disc'], emotion2,
                #                                            emo_loss_disc)
                # print('REAL_EMOTION2' + str(real_emotion_2))
                # # emo_disc_real_loss += emo_loss_disc(pred, torch.argmax(emotion2[0]))
                # #
                # # emo_loss_together = (disc_loss_1 + disc_loss_2) / 2
                # # losses['emo_disc_loss'] += emo_loss_together.item()
                # # emo_loss_together.backward()
                # # optimizer['emo_dis'].step()
                # # freezeNet(model['emo_disc'])

                ## Remove all gradients before Training disc and emoDisc
                optimizer['disc'].zero_grad()
                # optimizer['emo_dis'].zero_grad()

                pred = model['disc'](gtBatch)
                disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
                losses['disc_real_g'] += disc_real_loss.item()

                pred = model['disc'](x1e1.detach())
                disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                losses['disc_fake_g'] += disc_fake_loss.item()
                disc_loss = disc_real_loss + disc_fake_loss
                #
                #
                disc_loss.backward()
                # # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer['disc'].step()

                # print('----------------------------%step:{}---------------------'.format(global_step))

                #
                if global_step % (args.ckpt_interval / 2) == 0:
                    save_sample_images(inpBatch, x1e1, gtBatch, global_step,
                                       args.checkpoint_dir)
            except RuntimeError as e:
                inpImg1, spectrogram1, idx1, gtImg1, trgt1, prev_trg1, tlen1, ntoken1, padding_mask_1, pickedimg1, imgs1, = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                inpImg2, spectrogram2, idx2, gtImg2, trgt2, prev_trg2, tlen2, ntoken2, padding_mask_2, pickedimg2, imgs2, = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                perceptual_loss, l1loss, loss, disc_real_loss, disc_fake_loss, disc_loss, lip_loss, local_sync = 0, 0, 0, 0, 0, 0, 0, 0
                torch.cuda.empty_cache()
                print('TIMEOUT!!!')
                if "out of memory" in str(e):
                    print("CUDA out of memory error. Try reducing batch size or model size.")
                    oom_count += 1
                else:
                    raise e

            global_step += 1
            print('----------------------------%step:{}---------------------'.format(global_step))

            if global_step == 1 or global_step % args.ckpt_interval == 0:
                models = {
                    'gen': model['gen'],
                    'emoGen': model['emoGen'],
                    'decoder': model['decoder']
                }
                save_checkpoint(model['gen'], optimizer['gen'], global_step, args.checkpoint_dir, global_step
                                )
                save_checkpoint(model['emoGen'], optimizer['emo_gen'], global_step, args.checkpoint_dir, global_step,
                                prefix='emoGen_')
                save_checkpoint(model['decoder'], optimizer['decoder'], global_step, args.checkpoint_dir, global_step,
                                prefix='decoder_')
                save_checkpoint(model['disc'], optimizer['disc'], global_step, args.checkpoint_dir, global_step,
                                prefix='disc_')
                # save_checkpoint(model['emo_disc'], optimizer['emo_dis'], global_step, args.checkpoint_dir, global_step,
                #                 prefix='emo_disc_')
                # save_checkpoint(model['emo_disc'], optimizer['emo_opt'], global_step, args.checkpoint_dir, epoch, prefix='emo_disc')

            train_log = 'Train step: {} '.format(global_step)
            # if global_step % 100 == 0:
            #     return
            for key, value in losses.items():
                train_log += '{}: {:.4f} '.format(key, value / (step + 1))
            print(train_log)
            # train_log += '| gpu: {} | lr: {}'.format(get_gpu_memory_map()[args.gpu],
            #                                          optimizer['gen'].param_groups[0]['lr'])
            #
            # if global_step % args.ckpt_interval == 0:
            #     with torch.no_grad():
            #         average_sync_loss, valid_log = eval_model(data_loader['test'], avhubert, criterion, global_step,
            #                                                   device, model['gen'], model['disc'], args.cont_w,
            #                                                   recon_loss)
            #         prog_bar.set_description(valid_log)
            #
            #         logger.info(train_log)
            #         logger.info(valid_log)
            #         logger.info('\n')
            #
            #         status.update(average_sync_loss)
            #         stage, changed = status.check_status()
            #
            #         if changed and stage == 1:
            #             model['gen'].ft = True
            #             logger.info('Audio encoder start to finetune')
            #             logger.info('\n')
            #
            #         if changed and stage == 2:
            #             lip_train = True
            #             logger.info('Lip reading start to work')
            #             logger.info('\n')
            #
            #         if changed and stage == 3:
            #             logger.info('Training done')
            #             import sys
            #             sys.exit()
            #
            # prog_bar.set_description(train_log)
        logger.info('There are {} cases of out of memory in one epoch'.format(oom_count))


def train(device, model, avhubert, criterion, data_loader, optimizer, args, global_step, logger):
    print('Starting Step: {}'.format(global_step))

    lip_train = True
    emo_train = True
    model['gen'].ft = False
    status = status_manager(5)
    recon_loss = nn.L1Loss()
    l1_loss = nn.L1Loss()
    emo_loss_disc = nn.CrossEntropyLoss()
    cnt = 0

    accumulation_steps = 4

    # ((inpBatch_1, audioBatch_1, audio_idx_1, gt_batch_1, targetBatch_1, padding_mask_1, pickedimg_1, videoBatch_1,
    #   bbxs_1, emotion1, sample_len_1, sample_name_1),
    #  (inpBatch_2, audioBatch_2, audio_idx_2, gt_batch_2, targetBatch_2, padding_mask_2, pickedimg_2, videoBatch_2,
    #   bbxs_2, emotion2, sample_len_2, sample_name_2))
    for epoch in range(args.n_epoch):
        losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'emo_loss': 0, 'emo_cont_loss': 0, 'prec_g': 0, 'disc_real_g': 0,
                  'disc_fake_g': 0, 'emo_disc_loss': 0, 'c_l1': 0}
        prog_bar = tqdm(enumerate(data_loader['train']))
        print(len(data_loader['train']))
        oom_count = 0
        for step, (
                inpImg1, spectrogram1, idx1, gtImg1, ((trgt1, prev_trg1), tlen1, ntoken1), padding_mask_1, pickedimg1,
                imgs1,
                bbxs1, emotion1, sample_len_1, sample1,
                inpImg2, spectrogram2, idx2, gtImg2, ((trgt2, prev_trg2), tlen2, ntoken2), padding_mask_2, pickedimg2,
                imgs2,
                bbxs2, emotion2, sample_len_2,
                sample2) in prog_bar:
            for key in model.keys():
                model[key].train()
            print('SAMPLE1')
            print(sample1)
            print('SAMPLE2')
            print(sample2)

            criterion.report_accuracy = False
            padding_mask_1 = padding_mask_1.to(device)
            padding_mask_2 = padding_mask_2.to(device)
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
            trgt1, prev_trg1 = trgt1.to(device), prev_trg1.to(device)
            trgt2, prev_trg2 = trgt2.to(device), prev_trg2.to(device)
            # spectrogram1 = spectrogram1.transpose(1, 2)
            inpImg1, spectrogram1, gtImg1 = inpImg1.to(device), spectrogram1.to(
                device), gtImg1.to(device)
            # spectrogram2 = spectrogram2.transpose(1, 2)
            inpImg2, spectrogram2, gtImg2 = inpImg2.to(device), spectrogram2.to(
                device), gtImg2.to(device)

            # inpim, gtim = inpim.to(device), gtim.to(device)
            # if inpim.shape[0] < 120:
            #     continue
            prev_trg1, prev_trg2 = prev_trg1.to(device), prev_trg2.to(device)
            # trgt, prev_trg = trgt.to(device), prev_trg.to(device)
            # spectrogram, padding_mask = spectrogram.to(device), padding_mask.to(device)
            emotion1 = emotion1.to(device)
            emotion2 = emotion2.to(device)
            freezeNet(model['disc'])
            freezeNet(model['emo_disc'])
            for key in optimizer.keys():
                optimizer[key].zero_grad()

            net_input1 = {'source': {'audio': spectrogram1, 'video': None}, 'padding_mask': padding_mask_1,
                          'prev_output_tokens': prev_trg1}
            sample1 = {'net_input': net_input1, 'target_lengths': tlen1, 'ntokens': ntoken1, 'target': trgt1,
                       'emotion': emotion1}

            net_input2 = {'source': {'audio': spectrogram2, 'video': None}, 'padding_mask': padding_mask_2,
                          'prev_output_tokens': prev_trg2}
            sample2 = {'net_input': net_input2, 'target_lengths': tlen2, 'ntokens': ntoken2, 'target': trgt2,
                       'emotion': emotion2}

            print('------------------------------------------------')

            ###    {'gen': imGen, 'emoGen': emGen, 'decoder': decoder, 'disc': imDisc, 'emo_disc': emoDisc},
            ###    {'gen': gen_optimizer, 'emo_gen': emo_optimizer, 'decoder':decoder_optimizer, 'disc': disc_optimizer, 'emo_dis': emo_dis_optimizer},

            try:
                # optimizer['gen'].zero_grad()
                ##imgGen
                # x = model['gen']()
                x1, x1_embad = model['gen'](sample1, idx1)  # g: T*3*96*96
                # x1_1 = copy.deepcopy(x1)
                feats1, e1 = model['emoGen'](emotion1, inpImg1)
                feats1_1 = []
                for item in feats1:
                    feats1_1.append(item.clone())
                x2, x2_embad = model['gen'](sample2, idx2)
                # x2_1 = copy.deepcopy(x2)
                feats2, e2 = model['emoGen'](emotion2, inpImg2)
                feats2_1 = []
                for item in feats2:
                    feats2_1.append(item.clone())

                # print('XXXXXXXXXXXXX')
                # print(x1)
                # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
                # print(feats1)
                # print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                # print(e1)
                expanded_tensor = e1.unsqueeze(1)
                # 使用 torch.repeat_interleave() 函数复制每个样本的情绪嵌入对应次数
                repeated_tensor = torch.repeat_interleave(e1, torch.tensor(sample_len_1).to(device), dim=0).to(device)
                repeated_tensor1_2 = torch.repeat_interleave(e2, torch.tensor(sample_len_1).to(device), dim=0).to(
                    device)

                # 使用 torch.repeat_interleave() 函数复制每个样本的情绪嵌入对应次数
                repeated_tensor_2 = torch.repeat_interleave(e2, torch.tensor(sample_len_2).to(device), dim=0).to(device)
                repeated_tensor2_1 = torch.repeat_interleave(e1, torch.tensor(sample_len_2).to(device), dim=0).to(
                    device)

                ##生成face
                x1e1 = model['decoder'](x1, feats1, repeated_tensor, len(inpImg1.size()), spectrogram1.shape[0])
                x1e2 = model['decoder'](x1, feats1_1, repeated_tensor1_2, len(inpImg1.size()), spectrogram1.shape[0])
                x2e1 = model['decoder'](x2, feats2, repeated_tensor2_1, len(inpImg2.size()), spectrogram2.shape[0])
                x2e2 = model['decoder'](x2, feats2_1, repeated_tensor_2, len(inpImg2.size()), spectrogram2.shape[0])

                ## train real_picture discriminator
                # print(idx1.cpu().numpy().tolist())
                # print('pickedimg', pickedimg1)
                # id1 = idx1.cpu().numpy().tolist()
                # bbxs11 = torch.cat((bbxs1[0], bbxs1[1]))
                # print(bbxs11)
                # picks = torch.cat((pickedimg1[0], pickedimg1[1]))
                # imgs1_1 = torch.cat((imgs1[0], imgs1[1]))
                #
                # x1e1_processed_img = emb_roi2im([picks], [imgs1_1], [bbxs11], x1e1.cpu(), 'cpu')
                # print('x1e1''shape')
                # print(len(x1e1_processed_img))
                # print(x1e1_processed_img[0].shape)
                # for j, im in enumerate(x1e1_processed_img[0]):
                #     im = im.cpu().clone().detach().numpy().astype(np.uint8)
                #     cv2.imwrite('{}/{}.jpg'.format('/home/star/wy', j), im)

                ##train emo_disc

                ##train generator
                if lip_train:
                    processed_img = images2avhubert(pickedimg1, imgs1, bbxs1, x1e1, spectrogram1.shape[2], device)
                    processed_img_2 = images2avhubert(pickedimg2, imgs2, bbxs2, x2e2, spectrogram2.shape[2], device)
                    sample1['net_input']['source']['video'] = processed_img
                    sample1['net_input']['source']['audio'] = None
                    sample2['net_input']['source']['video'] = processed_img_2
                    sample2['net_input']['source']['audio'] = None
                    lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample1)
                    lip_loss_2, sample_size_2, logs_2, enc_out_2 = criterion(avhubert, sample2)
                    losses['lip'] += lip_loss.item()
                    losses['lip'] += lip_loss_2.item()
                    lip_loss += lip_loss_2
                    if args.cont_w > 0:
                        pickedVid, pickedAud = local_sync_loss(idx1, x1_embad, enc_out['encoder_out'])
                        local_sync = model['gen'].sync_net(torch.squeeze(pickedVid), torch.squeeze(pickedAud))
                        pickedVid_2, pickedAud_2 = local_sync_loss(idx2, x2_embad, enc_out_2['encoder_out'])
                        local_sync_2 = model['gen'].sync_net(torch.squeeze(pickedVid_2), torch.squeeze(pickedAud_2))
                        losses['local_sync'] += local_sync.item()
                        losses['local_sync'] += local_sync_2.item()
                        local_sync += local_sync_2
                    else:
                        local_sync = 0.
                else:
                    lip_loss, local_sync = 0., 0.

                if args.perp_w > 0.:
                    perceptual_loss = model['disc'].perceptual_forward(x1e1)
                    perceptual_loss += model['disc'].perceptual_forward(x1e2)
                    perceptual_loss += model['disc'].perceptual_forward(x2e1)
                    perceptual_loss += model['disc'].perceptual_forward(x2e2)
                    losses['prec_g'] += perceptual_loss.item()  ##gan
                else:
                    perceptual_loss = 0.
                #
                if args.e_w > 0:
                    loss1, x1emotion1 = get_emotions(x1e1, sample_len_1, model['emo_disc'], emotion1, emo_loss_disc)
                    loss2, x2emotion2 = get_emotions(x2e2, sample_len_2, model['emo_disc'], emotion2, emo_loss_disc)
                    emo_loss = loss1 + loss2

                    loss3, x2emotion1 = get_emotions(x2e1, sample_len_2, model['emo_disc'], emotion1, emo_loss_disc)
                    print('X2EMOTION1' + str(x2emotion1))
                    print('E1' + str(emotion1))
                    loss4, x1emotion2 = get_emotions(x1e2, sample_len_1, model['emo_disc'], emotion2, emo_loss_disc)
                    print('X1EMOTION2' + str(x1emotion2))
                    print('E2' + str(emotion2))

                    # emo对比损失
                    emo_loss += loss3 + loss4
                    emo_loss /= 4
                    # emo_cont_loss = emo_contrastive_loss(x1emotion1, x2emotion1, x1emotion2, x2emotion2)
                    emo_cont_loss = 0
                    #
                    losses['emo_loss'] += emo_loss.item()
                    # losses['emo_cont_loss'] += emo_cont_loss.item()

                l1loss = recon_loss(x1e1, gtImg1)
                l1loss += recon_loss(x2e2, gtImg2)
                losses['l1'] += l1loss.item()

                ##contro_loss
                # c_l1loss = 0.0
                if args.c_l1_w > 0:
                    if x1e2.shape[0] < gtImg2.shape[0]:
                        c_l1loss = recon_loss(x1e2, gtImg2[:int(x1e2.shape[0]), :, :, :])
                    elif x1e2.shape[0] > gtImg2.shape[0]:
                        c_l1loss = recon_loss(x1e2[: int(gtImg2.shape[0]), :, :, :], gtImg2)
                    else:
                        c_l1loss = recon_loss(x1e2, gtImg2)

                    if x2e1.shape[0] < gtImg1.shape[0]:
                        c_l1loss += recon_loss(x2e1, gtImg1[:int(x2e1.shape[0]), :, :, :])
                    elif x2e1.shape[0] > gtImg1.shape[0]:
                        c_l1loss += recon_loss(x2e1[: int(gtImg1.shape[0]), :, :, :], gtImg1)
                    else:
                        c_l1loss += recon_loss(x2e1, gtImg1)
                    losses['c_l1'] += c_l1loss.item()
                else:
                    c_l1loss = 0.0
                    # c_l1loss = recon_loss(x1e2, gtImg2)
                    # c_l1loss += recon_loss(x2e1, gtImg1)

                #
                loss = args.lip_w * lip_loss + args.perp_w * perceptual_loss + (
                        1. - args.lip_w - args.perp_w - args.e_w) * l1loss + args.cont_w * local_sync + args.e_w * emo_loss
                + args.e_c * emo_cont_loss + args.c_l1_w * c_l1loss
                # loss = loss / accumulation_steps
                loss.backward()

                # if (step + 1) % accumulation_steps == 0:
                optimizer['gen'].step()
                optimizer['gen'].zero_grad()
                optimizer['emo_gen'].step()
                optimizer['emo_gen'].zero_grad()
                optimizer['decoder'].step()
                optimizer['decoder'].zero_grad()
                # print('LOSSssssss')
                # print(loss)
                #
                # loss.backward()
                # optimizer['gen'].step()
                # optimizer['emo_gen'].step()
                # optimizer['decoder'].step()
                #
                # optimizer['gen'].zero_grad()
                # optimizer['emo_gen'].zero_grad()
                # optimizer['decoder'].zero_grad()

                unfreezeNet(model['disc'])
                # unfreezeNet(model['emo_disc'])
                disc_loss_1, real_emotion_1 = get_emotions(gtImg1, sample_len_1, model['emo_disc'], emotion1,
                                                           emo_loss_disc)
                print('REAL_EMOTION1' + str(real_emotion_1))

                # emo_disc_real_loss = emo_loss_disc(pred, torch.argmax(emotion1[0]))
                disc_loss_2, real_emotion_2 = get_emotions(gtImg2, sample_len_2, model['emo_disc'], emotion2,
                                                           emo_loss_disc)
                print('REAL_EMOTION2' + str(real_emotion_2))
                # emo_disc_real_loss += emo_loss_disc(pred, torch.argmax(emotion2[0]))
                #
                # emo_loss_together = (disc_loss_1 + disc_loss_2) / 2
                # losses['emo_disc_loss'] += emo_loss_together.item()
                # emo_loss_together.backward()
                # optimizer['emo_dis'].step()
                # freezeNet(model['emo_disc'])

                ## Remove all gradients before Training disc and emoDisc
                optimizer['disc'].zero_grad()
                # optimizer['emo_dis'].zero_grad()

                pred = model['disc'](torch.squeeze(gtImg1))
                disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
                pred = model['disc'](torch.squeeze(gtImg2))
                disc_real_loss += F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
                losses['disc_real_g'] += disc_real_loss.item() / 2

                pred = model['disc'](x1e1.detach())
                disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                pred = model['disc'](x1e2.detach())
                disc_fake_loss += F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                pred = model['disc'](x2e1.detach())
                disc_fake_loss += F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                pred = model['disc'](x2e2.detach())
                disc_fake_loss += F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                losses['disc_fake_g'] += disc_fake_loss.item() / 4
                disc_loss = disc_real_loss / 2 + disc_fake_loss / 4
                #
                #
                disc_loss.backward()
                # # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer['disc'].step()

                # print('----------------------------%step:{}---------------------'.format(global_step))

                #
                if global_step % (args.ckpt_interval / 2) == 0:
                    save_sample_images(inpImg1, x1e1, gtImg1, global_step,
                                       args.checkpoint_dir)
            except RuntimeError as e:
                inpImg1, spectrogram1, idx1, gtImg1, trgt1, prev_trg1, tlen1, ntoken1, padding_mask_1, pickedimg1, imgs1, = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                inpImg2, spectrogram2, idx2, gtImg2, trgt2, prev_trg2, tlen2, ntoken2, padding_mask_2, pickedimg2, imgs2, = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                perceptual_loss, l1loss, loss, disc_real_loss, disc_fake_loss, disc_loss, lip_loss, local_sync = 0, 0, 0, 0, 0, 0, 0, 0
                torch.cuda.empty_cache()
                print('TIMEOUT!!!')
                if "out of memory" in str(e):
                    print("CUDA out of memory error. Try reducing batch size or model size.")
                    oom_count += 1
                else:
                    raise e

            cnt += 1

            global_step += 1
            print('----------------------------%step:{}---------------------'.format(global_step))

            if global_step == 1 or global_step % args.ckpt_interval == 0:
                models = {
                    'gen': model['gen'],
                    'emoGen': model['emoGen'],
                    'decoder': model['decoder']
                }
                save_checkpoint(model['gen'], optimizer['gen'], global_step, args.checkpoint_dir, global_step
                                )
                save_checkpoint(model['emoGen'], optimizer['emo_gen'], global_step, args.checkpoint_dir, global_step,
                                prefix='emoGen_')
                save_checkpoint(model['decoder'], optimizer['decoder'], global_step, args.checkpoint_dir, global_step,
                                prefix='decoder_')
                save_checkpoint(model['disc'], optimizer['disc'], global_step, args.checkpoint_dir, global_step,
                                prefix='disc_')
                # save_checkpoint(model['emo_disc'], optimizer['emo_dis'], global_step, args.checkpoint_dir, global_step,
                #                 prefix='emo_disc_')
                # save_checkpoint(model['emo_disc'], optimizer['emo_opt'], global_step, args.checkpoint_dir, epoch, prefix='emo_disc')

            train_log = 'Train step: {} '.format(global_step)
            # if cnt % 5 == 0:
            #     return
            for key, value in losses.items():
                train_log += '{}: {:.4f} '.format(key, value / (step + 1))
            print(train_log)
            # train_log += '| gpu: {} | lr: {}'.format(get_gpu_memory_map()[args.gpu],
            #                                          optimizer['gen'].param_groups[0]['lr'])
            #
            # if global_step % args.ckpt_interval == 0:
            #     with torch.no_grad():
            #         average_sync_loss, valid_log = eval_model(data_loader['test'], avhubert, criterion, global_step,
            #                                                   device, model['gen'], model['disc'], args.cont_w,
            #                                                   recon_loss)
            #         prog_bar.set_description(valid_log)
            #
            #         logger.info(train_log)
            #         logger.info(valid_log)
            #         logger.info('\n')
            #
            #         status.update(average_sync_loss)
            #         stage, changed = status.check_status()
            #
            #         if changed and stage == 1:
            #             model['gen'].ft = True
            #             logger.info('Audio encoder start to finetune')
            #             logger.info('\n')
            #
            #         if changed and stage == 2:
            #             lip_train = True
            #             logger.info('Lip reading start to work')
            #             logger.info('\n')
            #
            #         if changed and stage == 3:
            #             logger.info('Training done')
            #             import sys
            #             sys.exit()
            #
            # prog_bar.set_description(train_log)
        logger.info('There are {} cases of out of memory in one epoch'.format(oom_count))


def eval_model(test_data_loader, criterion, global_step, device, model, disc, emo_disc):
    print('Evaluating after training of {} steps'.format(global_step))
    n_correct, n_total = 0, 0
    losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'prec_g': 0, 'disc_real_g': 0, 'disc_fake_g': 0}

    correct_nums = [0, 0, 0, 0, 0, 0]
    total_nums = [0, 0, 0, 0, 0, 0]
    correct_arc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for epoch in range(args.n_epoch):
        model['gen'].eval()
        model['decoder'].eval()
        model['emoGen'].eval()
        model['disc'].eval()
        model[emo_disc].eval()
        disc.eval()
        # with torch.no_grad():
        #     try:
        #         for step, (
        #                 inpImg1, spectrogram1, idx1, gtImg1, ((trgt1, prev_trg1), tlen1, ntoken1), padding_mask_1,
        #                 pickedimg1,
        #                 imgs1,
        #                 bbxs1, emotion1, sample_len_1, sample1,
        #                 inpImg2, spectrogram2, idx2, gtImg2, ((trgt2, prev_trg2), tlen2, ntoken2), padding_mask_2,
        #                 pickedimg2,
        #                 imgs2,
        #                 bbxs2, emotion2, sample_len_2,
        #                 sample2) in enumerate(
        #             (test_data_loader)):
        #
        #             print('SAMPLE1')
        #             print(sample1)
        #             print('SAMPLE2')
        #             print(sample2)
        #
        #             criterion.report_accuracy = False
        #             padding_mask_1 = padding_mask_1.to(device)
        #             padding_mask_2 = padding_mask_2.to(device)
        #             idx1 = idx1.to(device)
        #             idx2 = idx2.to(device)
        #             trgt1, prev_trg1 = trgt1.to(device), prev_trg1.to(device)
        #             trgt2, prev_trg2 = trgt2.to(device), prev_trg2.to(device)
        #             # spectrogram1 = spectrogram1.transpose(1, 2)
        #             inpImg1, spectrogram1, gtImg1 = inpImg1.to(device), spectrogram1.to(
        #                 device), gtImg1.to(device)
        #             # spectrogram2 = spectrogram2.transpose(1, 2)
        #             inpImg2, spectrogram2, gtImg2 = inpImg2.to(device), spectrogram2.to(
        #                 device), gtImg2.to(device)
        #
        #             # inpim, gtim = inpim.to(device), gtim.to(device)
        #             # if inpim.shape[0] < 120:
        #             #     continue
        #             prev_trg1, prev_trg2 = prev_trg1.to(device), prev_trg2.to(device)
        #             # trgt, prev_trg = trgt.to(device), prev_trg.to(device)
        #             # spectrogram, padding_mask = spectrogram.to(device), padding_mask.to(device)
        #             emotion1 = emotion1.to(device)
        #             emotion2 = emotion2.to(device)
        #             freezeNet(model['disc'])
        #             freezeNet(model['emo_disc'])
        #             for key in optimizer.keys():
        #                 optimizer[key].zero_grad()
        #
        #             net_input1 = {'source': {'audio': spectrogram1, 'video': None}, 'padding_mask': padding_mask_1,
        #                           'prev_output_tokens': prev_trg1}
        #             sample1 = {'net_input': net_input1, 'target_lengths': tlen1, 'ntokens': ntoken1, 'target': trgt1,
        #                        'emotion': emotion1}
        #
        #             net_input2 = {'source': {'audio': spectrogram2, 'video': None}, 'padding_mask': padding_mask_2,
        #                           'prev_output_tokens': prev_trg2}
        #             sample2 = {'net_input': net_input2, 'target_lengths': tlen2, 'ntokens': ntoken2, 'target': trgt2,
        #                        'emotion': emotion2}
        #
        #             print('------------------------------------------------')
        #
        #             ###    {'gen': imGen, 'emoGen': emGen, 'decoder': decoder, 'disc': imDisc, 'emo_disc': emoDisc},
        #             ###    {'gen': gen_optimizer, 'emo_gen': emo_optimizer, 'decoder':decoder_optimizer, 'disc': disc_optimizer, 'emo_dis': emo_dis_optimizer},
        #
        #             try:
        #                 # optimizer['gen'].zero_grad()
        #                 ##imgGen
        #                 # x = model['gen']()
        #                 x1, x1_embad = model['gen'](sample1, idx1)  # g: T*3*96*96
        #                 # x1_1 = copy.deepcopy(x1)
        #                 feats1, e1 = model['emoGen'](emotion1, inpImg1)
        #                 feats1_1 = []
        #                 for item in feats1:
        #                     feats1_1.append(item.clone())
        #                 x2, x2_embad = model['gen'](sample2, idx2)
        #                 feats2, e2 = model['emoGen'](emotion2, inpImg2)
        #                 feats2_1 = []
        #                 for item in feats2:
        #                     feats2_1.append(item.clone())
        #                 # 使用 torch.repeat_interleave() 函数复制每个样本的情绪嵌入对应次数
        #                 repeated_tensor = torch.repeat_interleave(e1, torch.tensor(sample_len_1).to(device), dim=0).to(
        #                     device)
        #                 repeated_tensor1_2 = torch.repeat_interleave(e2, torch.tensor(sample_len_1).to(device),
        #                                                              dim=0).to(
        #                     device)
        #
        #                 # 使用 torch.repeat_interleave() 函数复制每个样本的情绪嵌入对应次数
        #                 repeated_tensor_2 = torch.repeat_interleave(e2, torch.tensor(sample_len_2).to(device),
        #                                                             dim=0).to(device)
        #                 repeated_tensor2_1 = torch.repeat_interleave(e1, torch.tensor(sample_len_2).to(device),
        #                                                              dim=0).to(
        #                     device)
        #
        #                 x1e1 = model['decoder'](x1, feats1, repeated_tensor, len(inpImg1.size()), spectrogram1.shape[0])
        #                 x1e2 = model['decoder'](x1, feats1_1, repeated_tensor1_2, len(inpImg1.size()),
        #                                         spectrogram1.shape[0])
        #                 x2e1 = model['decoder'](x2, feats2, repeated_tensor2_1, len(inpImg2.size()),
        #                                         spectrogram2.shape[0])
        #                 x2e2 = model['decoder'](x2, feats2_1, repeated_tensor_2, len(inpImg2.size()),
        #                                         spectrogram2.shape[0])
        #
        #
        #
        #
        #
        #
        #             del x, spectrogram, gt, trgt, prev_trg, padding_mask, emotion, net_input, sample, g, enc_audio
        #             gc.collect()
        #             torch.cuda.empty_cache()
        #             global_step += 1
        #             if global_step % 1000 == 0 and global_step != 0:
        #                 for i in range(len(total_nums)):
        #                     correct_arc[i] = (float)(correct_nums[i] / total_nums[i])
        #                 print('step:{}, arc:{}'.format(step, correct_arc))
        #                 with open('correct_arc_step{}.txt'.format(step), 'a') as file:
        #                     file.write('step {}: {}\n'.format(step, correct_arc))
        #         print('arc:{}'.format(correct_arc))
        #         break
        #     except ValueError:
        #         continue

    # processed_img = images2avhubert(vidx, videos, bbxs, g, spectrogram.shape[2], device)
    # sample['net_input']['source']['video'] = processed_img
    # sample['net_input']['source']['audio'] = None
    #
    # lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample)
    # losses['lip'] += lip_loss.item()

    # if cont_w > 0:
    #     pickedVid, pickedAud = local_sync_loss(audio_idx, enc_audio, enc_out['encoder_out'])
    #     local_sync = model.sync_net(pickedVid, pickedAud)
    #     losses['local_sync'] += local_sync.item()
    #
    # n_correct += logs['n_correct']
    # n_total += logs['total']
    # if args.perp_w > 0.:
    #     perceptual_loss = disc.perceptual_forward(g)
    #     losses['prec_g'] += perceptual_loss.item()
    #
    # l1loss = recon_loss(g, gt)
    # losses['l1'] += l1loss.item()

    # avewer = 1 - n_correct / n_total

    # valid_log = 'Valid step: {} '.format(global_step)
    # for key, value in losses.items():
    #     valid_log += '{}: {:.4f} '.format(key, value / (step + 1))
    # valid_log += '| wer: {}'.format(avewer)
    # print(valid_log)
    # return losses['l1'], valid_log


def save_checkpoint(model, optimizer, global_step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


def save_models_checkpoint(models, optimizer, global_step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    torch.save({
        "state_dict": {name: model.state_dict() for name, model in models.items()},
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint_models(filename, models, optimizer):
    checkpoint = _load(filename)
    for name, model in models.items():
        model.load_state_dict(checkpoint['models_state_dict'][name])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['global_step']
    return epoch


def load_checkpoint(path, model, optimizer, logger, reset_optimizer=False, overwrite_global_states=True):
    print("Load checkpoint from: {}".format(path))
    logger.info("Load checkpoint from: {}".format(path))
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
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
    else:
        global_step = 0

    return global_step


# def init_weights(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
#         torch.nn.init.xavier_uniform_(m.weight)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Code to train the Wav2Lip model WITH the visual quality discriminator')

    # dataset
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument("--audio_root", help="Root folder of audio", required=True, type=str)
    parser.add_argument("--word_root", help="Root folder of audio", required=False, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--file_dir", help="Root folder of filelists", required=True, type=str)
    parser.add_argument("--file_dir_2", help="Root folder of filelists", required=True, type=str)
    parser.add_argument('--batch_size', help='batch size of training', default=2, type=int)
    parser.add_argument('--batch_size_ref', help='batch size of training_ref', default=8, type=int)
    parser.add_argument('--num_worker', help='number of worker', default=6, type=int)

    # checkpoint loading and saving
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--gen_checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
    parser.add_argument('--disc_checkpoint_path', help='Resume discriminator from this checkpoint', default=None,
                        type=str)
    parser.add_argument('--emo_disc_checkpoint_path', help='Resume emo_discriminator from this checkpoint',
                        default=None,
                        type=str)
    parser.add_argument('--emoGen_checkpoint_path', help='Resume emo_discriminator from this checkpoint',
                        default=None,
                        type=str)
    parser.add_argument('--decoder_checkpoint_path', help='Resume emo_discriminator from this checkpoint',
                        default=None,
                        type=str)
    parser.add_argument('--avhubert_path', help='Resume avhubert from this checkpoint', default=None, type=str)
    parser.add_argument('--avhubert_root', help='Path of av_hubert root', required=True, type=str)

    # optimizer
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)

    # loss
    parser.add_argument('--lip_w', help='weight of lip-reading expert', default=1e-6, type=float)
    parser.add_argument('--cont_w', help='weight of contrastive learning', default=1e-4, type=float)
    parser.add_argument('--perp_w', help='weight of perceptual loss', default=0.07, type=float)
    parser.add_argument('--e_w', help='weight of emotion loss', default=0.01, type=float)
    parser.add_argument('--e_w_ref', help='weight of emotion loss', default=0.000, type=float)
    parser.add_argument('--e_c', help='weight of emotion_contrastive loss', default=0, type=float)
    parser.add_argument('--c_l1_w', help='weight of contrastive_l1 loss', default=0.00, type=float)
    ##batch_size
    # training
    parser.add_argument('--gpu', help='index of gpu used', default=0, type=int)
    parser.add_argument('--n_epoch', help='number of epoch', default=100000000, type=int)
    parser.add_argument('--log_name', help='name of a log file', default='emotalklip', type=str)
    parser.add_argument('--ckpt_interval', help='The interval of saving a checkpoint', default=500, type=int)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print('use_cuda: {}, {}'.format(args.gpu, use_cuda))
    device = "cuda:{}".format(args.gpu) if use_cuda else "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # torch.cuda.device_count()

    avhubert, label_proc, generator, criterion, encoder = retrieve_avhubert(args.avhubert_root, args.avhubert_path,
                                                                            device)

    # Dataset and Dataloader setup
    train_dataset = Talklipdata('filename', args, label_proc, True)
    # train_dataset_ref = Talklipdata_2('train', args, label_proc)
    # train_dataset_re = Talklipdata('lrw_files', args, label_proc, True)
    # test_dataset = Talklipdata('valid', args, label_proc)

    # train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
    #                                num_workers=args.num_worker)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_worker, collate_fn=collate_fn)
    # train_data_re_loader = DataLoader(train_dataset_ref, batch_size=args.batch_size_ref, shuffle=True,
    #                                   num_workers=args.num_worker, collate_fn=collate_fn_re)

    # test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
    #                               num_workers=args.num_worker)

    imGen = TalkLip(encoder, 768).to(device)
    emGen = EmotionEncoder().to(device)
    imDisc = TalkLip_disc_qual().to(device)
    decoder = Decoder().to(device)
    emoDisc = DISCEMO().to(device)
    # for param in emoDisc.parameters():
    #     param.requires_grad_(True)
    # emoDisc.apply(init_weights)

    gen_optimizer = optim.Adam(list(imGen.parameters()) + list(emGen.parameters()) + list(decoder.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))

    optimizer = optim.Adam([p for p in imGen.parameters() if p.requires_grad],
                           lr=args.lr, betas=(0.5, 0.999))

    emo_optimizer = optim.Adam([p for p in emGen.parameters() if p.requires_grad],
                               lr=args.lr, betas=(0.5, 0.999))

    decoder_optimizer = optim.Adam([p for p in decoder.parameters() if p.requires_grad],
                                   lr=args.lr, betas=(0.5, 0.999))

    disc_optimizer = optim.Adam([p for p in imDisc.parameters() if p.requires_grad],
                                lr=args.lr, betas=(0.5, 0.999))

    emo_dis_optimizer = optim.Adam([p for p in emoDisc.parameters() if p.requires_grad], lr=1e-4, betas=(0.5, 0.999))
    # emo_optimizer = torch.optim.Adam([p for p in emoDisc.parameters() if p.requires_grad], lr=1e-06, betas=(0.5, 0.999))
    # emo_scheduler = torch.optim.lr_scheduler.StepLR(emo_optimizer, args.lr, gamma=0.1, last_epoch=-1)

    os.makedirs('log/', exist_ok=True)
    logger = init_logging(log_name='log/{}.log'.format(args.log_name))

    # 查看GPU使用情况
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())

    torch.cuda.empty_cache()

    global_step = 0
    models = {
        'gen': imGen,
        'emoGen': emGen,
        'decoder': decoder
    }
    if args.gen_checkpoint_path is not None:
        global_step = load_checkpoint(args.gen_checkpoint_path, imGen, optimizer, logger,
                                      reset_optimizer=False, overwrite_global_states=False)
        # global_step = load_checkpoint(args.gen_checkpoint_path, imGen, optimizer, logger)
    if args.emoGen_checkpoint_path is not None:
        global_step = load_checkpoint(args.emoGen_checkpoint_path, emGen, emo_optimizer, logger,
                                      reset_optimizer=False, overwrite_global_states=True)
        # global_step = load_checkpoint(args.gen_checkpoint_path, imGen, optimizer, logger)
    if args.disc_checkpoint_path is not None:
        global_step = load_checkpoint(args.disc_checkpoint_path, imDisc, disc_optimizer, logger,
                                      reset_optimizer=False, overwrite_global_states=True)
    if args.decoder_checkpoint_path is not None:
        global_step = load_checkpoint(args.decoder_checkpoint_path, decoder, decoder_optimizer, logger,
                                      reset_optimizer=False, overwrite_global_states=True)

    print(global_step)
    # if args.emo_disc_checkpoint_path is not None:
    #     load_checkpoint(args.emo_disc_checkpoint_path, emoDisc, emo_dis_optimizer, logger,
    #                     reset_optimizer=False, overwrite_global_states=False)
    # load_checkpoint('/ssd2/m3lab/usrs/wy/TalkLip/real_check/emo_disc_checkpoint_step000218000.pth', emoDisc,
    #                 emo_dis_optimizer, logger,
    #                 reset_optimizer=False, overwrite_global_states=False)
    # emoDisc.load_state_dict(torch.load('/ssd2/m3lab/usrs/wy/TalkLip/real_check/emo_disc_checkpoint_step000218000.pth'))
    # # if args.emo_disc_checkpoint_path is not None:
    # #     load_checkpoint(args.emo_disc_checkpoint_path, emoDisc, emo_optimizer, logger,
    # #                     reset_optimizer=False, overwrite_global_states=False)
    # # else:
    # emoDisc.load_state_dict(torch.load('/ssd2/m3lab/usrs/wy/EmoGen/emo-checkpoint/disc_emo_52000.pth'))
    load_checkpoint('/home/star/wy/emo_disc_ checkpoint/emo_disccheckpoint_step000062000.pth', emoDisc,
                    emo_dis_optimizer, logger,
                    reset_optimizer=False, overwrite_global_states=False)
    # load_checkpoint('/ssd2/m3lab/usrs/wy/TalkLip/vis_dis.pth', imDisc,
    #                 disc_optimizer, logger,
    #                 reset_optimizer=False, overwrite_global_states=False)
    # load_checkpoint('/home/star/wy/vis_dis.pth', imDisc,
    #                 disc_optimizer, logger,
    #                 reset_optimizer=False, overwrite_global_states=False)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    # test_data_loader, criterion, global_step, device, model, disc, emo_disc
    # eval_model(train_data_loader, criterion, 0, device, imGen, imDisc, emoDisc)
    # init_weights(emGen)
    # init_weights(device)
    # global_step = 124000

    while True:
        train(device, {'gen': imGen, 'emoGen': emGen, 'decoder': decoder, 'disc': imDisc, 'emo_disc': emoDisc},
              avhubert,
              criterion,
              {'train': train_data_loader, 'test': None},
              {'gen': optimizer, 'emo_gen': emo_optimizer, 'decoder': decoder_optimizer, 'disc': disc_optimizer,
               'emo_dis': emo_dis_optimizer},
              args,
              global_step, logger)
        # torch.cuda.empty_cache()
        # #
        # global_step += 400

        # train_ref(device, {'gen': imGen, 'emoGen': emGen, 'decoder': decoder, 'disc': imDisc, 'emo_disc': emoDisc},
        #           avhubert,
        #           criterion,
        #           {'train': train_data_re_loader, 'test': None},
        #           {'gen': optimizer, 'emo_gen': emo_optimizer, 'decoder': decoder_optimizer, 'disc': disc_optimizer,
        #            'emo_dis': emo_dis_optimizer},
        #           args,
        #           global_step, logger)
        # global_step += 100

        # torch.cuda.empty_cache()

    ##/ssd2/m3lab/usrs/wy/EmoGen/emo-checkpoint/disc_emo_52000.pth


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
