"""
This file is adapted from 'wav2lip' GitHub repository
https://github.com/Rudrabha/Wav2Lip.

We design a new audio encoder with transformer
"""

import contextlib
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .learn_sync import av_sync

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class TalkLip(nn.Module):
    def __init__(self, audio_encoder, audio_num, res_layers=None):
        super(TalkLip, self).__init__()

        enc_channel = [6, 16, 32, 64, 128, 256, 512, 512]

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(enc_channel[0], enc_channel[1], kernel_size=7, stride=1, padding=3)),  # 16, 96, 96

            nn.Sequential(Conv2d(enc_channel[1], enc_channel[2], kernel_size=3, stride=2, padding=1),  # 32, 48, 48
                          Conv2d(enc_channel[2], enc_channel[2], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[2], enc_channel[2], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[2], enc_channel[3], kernel_size=3, stride=2, padding=1),  # 64, 24,24
                          Conv2d(enc_channel[3], enc_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[3], enc_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[3], enc_channel[3], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[3], enc_channel[4], kernel_size=3, stride=2, padding=1),  # 128, 12,12
                          Conv2d(enc_channel[4], enc_channel[4], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[4], enc_channel[4], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[4], enc_channel[5], kernel_size=3, stride=2, padding=1),  # 256, 6,6
                          Conv2d(enc_channel[5], enc_channel[5], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(enc_channel[5], enc_channel[5], kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(enc_channel[5], enc_channel[6], kernel_size=3, stride=2, padding=1),  # 512, 3,3
                          Conv2d(enc_channel[6], enc_channel[6], kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(enc_channel[6], enc_channel[7], kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(enc_channel[7], enc_channel[7], kernel_size=1, stride=1, padding=0)), ])

        self.audio_encoder = audio_encoder
        self.audio_map = nn.Linear(audio_num, enc_channel[-1])

        self.emotion_encoder = nn.Sequential(
            nn.Linear(6, self.args.emo_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.args.emo_dim, self.args.emo_dim),
            nn.LeakyReLU(0.2)
        )

        dec_channel = [512, 512, 512, 384, 256, 128, 64]
        upsamp_channel = []
        if res_layers is None:
            self.res_layers = len(dec_channel)
        else:
            self.res_layers = res_layers
        for i in range(len(dec_channel)):
            if i < self.res_layers:
                upsamp_channel.append(enc_channel[-i - 1] + dec_channel[i])
            else:
                upsamp_channel.append(dec_channel[i])

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(enc_channel[-1], dec_channel[0], kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(upsamp_channel[0], dec_channel[1], kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(dec_channel[1], dec_channel[1], kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(upsamp_channel[1], dec_channel[2], kernel_size=3, stride=2, padding=1,
                                          output_padding=1),
                          Conv2d(dec_channel[2], dec_channel[2], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[2], dec_channel[2], kernel_size=3, stride=1, padding=1, residual=True), ),
            # 6, 6

            nn.Sequential(Conv2dTranspose(upsamp_channel[2], dec_channel[3], kernel_size=3, stride=2, padding=1,
                                          output_padding=1),
                          Conv2d(dec_channel[3], dec_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[3], dec_channel[3], kernel_size=3, stride=1, padding=1, residual=True), ),
            # 12, 12

            nn.Sequential(Conv2dTranspose(upsamp_channel[3], dec_channel[4], kernel_size=3, stride=2, padding=1,
                                          output_padding=1),
                          Conv2d(dec_channel[4], dec_channel[4], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[4], dec_channel[4], kernel_size=3, stride=1, padding=1, residual=True), ),
            # 24, 24

            nn.Sequential(Conv2dTranspose(upsamp_channel[4], dec_channel[5], kernel_size=3, stride=2, padding=1,
                                          output_padding=1),
                          Conv2d(dec_channel[5], dec_channel[5], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[5], dec_channel[5], kernel_size=3, stride=1, padding=1, residual=True), ),
            # 48, 48

            nn.Sequential(Conv2dTranspose(upsamp_channel[5], dec_channel[6], kernel_size=3, stride=2, padding=1,
                                          output_padding=1),
                          Conv2d(dec_channel[6], dec_channel[6], kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(dec_channel[6], dec_channel[6], kernel_size=3, stride=1, padding=1,
                                 residual=True), ), ])  # 96,96

        self.emotion_decoder = nn.Sequential(

        )

        self.output_block = nn.Sequential(Conv2d(upsamp_channel[6], 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())
        self.ft = False
        self.sync_net = av_sync(audio_num, audio_num)

    #             sample = {'net_input': net_input, 'target_lengths': tlen, 'ntokens': ntoken, 'target': trgt,
    #                       'emotion': emotion}
    #             net_input = {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask,
    #                          'prev_output_tokens': prev_trg}
    def forward(self, sample, face_sequences, idAudio, B):

        input_dim_size = len(face_sequences.size())

        # input 1*F*T
        with torch.no_grad() if not self.ft else contextlib.ExitStack():
            enc_out = self.audio_encoder(**sample["net_input"])
        # T*B*C, B*T

        audio_embedding, audio_padding = enc_out['encoder_out'], enc_out['padding_mask']
        emtion_in = sample['emotion'].unsqueeze(1).repeat(1, 5, 1)
        emotion_embedding = self.emotion_encoder(emtion_in)
        # ee_needed =  torch.mean(emotion_embedding,0).unsqueeze(0)
        emotion_embedding = emotion_embedding.view(-1, 512, 1, 1)  # B*T, 512, 1, 1

        feats = []
        x = face_sequences
        # output is N*512*1*1
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        # T*B*C -> N*C -> N*512*1*1

        x = audio_embedding.permute(1, 0, 2).reshape(-1, audio_embedding.shape[2])[idAudio]
        x = self.audio_map(x).reshape(x.shape[0], 512, 1, 1)
        x = torch.cat((x, emotion_embedding), dim=1)

        for i, f in enumerate(self.face_decoder_blocks):
            x = f(x)
            try:
                if i < self.res_layers:
                    x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs, audio_embedding  # , ys_hat#, wer

    def get_aud_emb(self, sample):
        with torch.no_grad():
            enc_out = self.audio_encoder(**sample["net_input"])
        # T*B*C, B*T

        audio_embedding, audio_padding = enc_out['encoder_out'], enc_out['padding_mask']
        return audio_embedding


class TalkLip_disc_qual(nn.Module):
    def __init__(self):
        super(TalkLip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12,12
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def perceptual_forward(self, false_face_sequences):
        """
        force discriminator output given generated faces as input to 1
        Args:
            false_face_sequences: T*C*H*W

        Returns:

        """
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                 torch.ones((len(false_feats), 1)).to(
                                                     false_face_sequences.device))  # .cuda()

        return false_pred_loss

    def forward(self, face_sequences):

        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)


class DISCEMO(nn.Module):
    def __init__(self, args, debug=False):
        super(DISCEMO, self).__init__()
        self.args = args
        self.drp_rate = 0

        self.filters = [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2)]

        prev_filters = 3
        for i, (num_filters, filter_size, stride) in enumerate(self.filters):
            setattr(self,
                    'conv_' + str(i + 1),
                    nn.Sequential(
                        nn.Conv2d(prev_filters, num_filters, kernel_size=filter_size, stride=stride,
                                  padding=filter_size // 2),
                        nn.LeakyReLU(0.3)
                    )
                    )
            prev_filters = num_filters

        self.projector = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.LeakyReLU(0.3),
            nn.Linear(2048, 512)
        )

        self.rnn_1 = nn.LSTM(512, 512, 1, bidirectional=False, batch_first=True)

        self.cls = nn.Sequential(
            nn.Linear(512, 6 + 1)
        )

        # Optimizer
        # self.opt = optim.Adam(list(self.parameters()), lr=self.args.lr_emo, betas=(0.5, 0.999))
        # self.opt = optim.RMSprop(list(self.parameters()), lr = params['LR_DE'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)
        self.lossE = nn.CrossEntropyLoss()
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.95, last_epoch=-1)

    def percep_forward(self, video, r_emotion):
        x = video
        n, t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.contiguous().view(t * n, c, w, h)
        for i in range(len(self.filters)):
            x = getattr(self, 'conv_' + str(i + 1))(x)
        h = x.view(n, t, -1)
        h = self.projector(h)

        h, _ = self.rnn_1(h)

        h_class = self.cls(h[:, -1, :])
        lossE = self.lossE(h_class, torch.argmax(r_emotion, dim=1))
        return lossE

    def forward(self, condition, video):
        x = video
        n, t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.contiguous().view(t * n, c, w, h)
        for i in range(len(self.filters)):
            x = getattr(self, 'conv_' + str(i + 1))(x)
        h = x.view(n, t, -1)
        h = self.projector(h)

        h, _ = self.rnn_1(h)

        h_class = self.cls(h[:, -1, :])

        return h_class

    def enableGrad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad_(requires_grad)

    def compute_grad_penalty(self, video_gt, video_pd, image_c, classes):
        interpolated = video_gt.data  # + (1-alpha) * video_pd.data
        interpolated = Variable(interpolated, requires_grad=True)

        d_out_c = self.forward(image_c, interpolated)
        classes = torch.cat((classes, torch.zeros(classes.size(0), 1).to(self.args.device)), 1)

        grad_dout = torch.autograd.grad(
            outputs=d_out_c,
            inputs=interpolated,
            grad_outputs=classes.to(self.args.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_dout = grad_dout.contiguous().view(grad_dout.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(grad_dout ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean(), gradients_norm.mean()
