"""
This file is adapted from 'wav2lip' GitHub repository
https://github.com/Rudrabha/Wav2Lip.

We design a new audio encoder with transformer
"""

import contextlib
import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .learn_sync import av_sync

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim, feat_dim):
        super(AttentionMechanism, self).__init__()
        self.query = nn.Linear(embed_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)

    def forward(self, emotion_embedding, face_feat):
        # 计算注意力得分
        query = self.query(emotion_embedding)  # [B, embed_dim] -> [B, feat_dim]
        key = self.key(face_feat)  # [B, T, feat_dim]
        value = self.value(face_feat)  # [B, T, feat_dim]

        attention_scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) / (face_feat.size(-1) ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attended_feat = torch.matmul(attention_weights, value)
        return attended_feat


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = F.relu(out)
        return out


class EmotionEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, latent_dim=512, num_residual_blocks=3):
        super(EmotionEncoder, self).__init__()

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

        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.InstanceNorm1d(hidden_dim)
        #
        # self.residual_blocks = nn.ModuleList(
        #     [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_residual_blocks)])
        #
        # self.fc2 = nn.Linear(hidden_dim, latent_dim)
        # self.bn2 = nn.InstanceNorm1d(latent_dim)
        # self.dropout = nn.Dropout(p=0.5)
        #
        # self.hidden_dim_link = 128
        # self.fc = nn.Linear(self.hidden_dim_link * 2, 1)
        self.emotion_encoder = nn.Sequential(
            nn.Linear(6, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.emotion_encoder_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(6, enc_channel[0]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[0], enc_channel[1]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[1], enc_channel[2]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[2], enc_channel[3]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[3], enc_channel[4]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[4], enc_channel[5]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[5], enc_channel[6]), nn.LeakyReLU(0.2)),
            nn.Sequential(nn.Linear(enc_channel[6], enc_channel[7]), nn.LeakyReLU(0.2)),
        ])

        # self.attention_layers = nn.ModuleList(
        #     [AttentionMechanism(latent_dim, face_feat_dim) for face_feat_dim in enc_channel[1:]])

    def forward(self, x, face_sequences):
        # x = F.relu(self.fc1(x))
        # for block in self.residual_blocks:
        #     x = block(x)
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.emotion_encoder(x)

        feats = []
        face = face_sequences
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        # output is N*512*1*1
        for f in self.face_encoder_blocks:
            face = f(face)
            feats.append(face)

        emotion_embedding = x.view(-1, 512, 1, 1)

        # combined_feats = []
        # for i, face_feat in enumerate(feats):
        #     face_feat = face_feat.permute(0, 2, 3, 1).contiguous()
        #     attended_feat = self.attention_layers[i](x, face_feat)
        #
        #     attended_feat = attended_feat.repeat(1, 1, attended_feat.shape[1], 1).permute(0, 3, 1, 2).contiguous()
        #     combined_feats.append(attended_feat)

        return feats, emotion_embedding


class TalkLip(nn.Module):
    def __init__(self, audio_encoder, audio_num, res_layers=None):
        super(TalkLip, self).__init__()

        self.audio_encoder = audio_encoder
        self.audio_map = nn.Linear(audio_num, 512)

        # self.emotion_encoder = nn.Sequential(
        #     nn.Linear(6, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2)
        # )
        # nn.Linear(6, 256),
        # nn.ReLU(True),
        # nn.Linear(256, 512),
        # nn.ReLU(True),

        # dec_channel = [1024, 512, 512, 384, 256, 128, 64]
        # upsamp_channel = []
        # if res_layers is None:
        #     self.res_layers = len(dec_channel)
        # else:
        #     self.res_layers = res_layers
        # for i in range(len(dec_channel)):
        #     if i < self.res_layers:
        #         upsamp_channel.append(enc_channel[-i - 1] + dec_channel[i])
        #     else:
        #         upsamp_channel.append(dec_channel[i])
        #
        # self.face_decoder_blocks = nn.ModuleList([
        #     nn.Sequential(Conv2d(1024, dec_channel[0], kernel_size=1, stride=1, padding=0), ),
        #
        #     nn.Sequential(Conv2dTranspose(upsamp_channel[0], dec_channel[1], kernel_size=3, stride=1, padding=0),  # 3,3
        #                   Conv2d(dec_channel[1], dec_channel[1], kernel_size=3, stride=1, padding=1, residual=True), ),
        #
        #     nn.Sequential(Conv2dTranspose(upsamp_channel[1], dec_channel[2], kernel_size=3, stride=2, padding=1,
        #                                   output_padding=1),
        #                   Conv2d(dec_channel[2], dec_channel[2], kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(dec_channel[2], dec_channel[2], kernel_size=3, stride=1, padding=1, residual=True), ),
        #     # 6, 6
        #
        #     nn.Sequential(Conv2dTranspose(upsamp_channel[2], dec_channel[3], kernel_size=3, stride=2, padding=1,
        #                                   output_padding=1),
        #                   Conv2d(dec_channel[3], dec_channel[3], kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(dec_channel[3], dec_channel[3], kernel_size=3, stride=1, padding=1, residual=True), ),
        #     # 12, 12
        #
        #     nn.Sequential(Conv2dTranspose(upsamp_channel[3], dec_channel[4], kernel_size=3, stride=2, padding=1,
        #                                   output_padding=1),
        #                   Conv2d(dec_channel[4], dec_channel[4], kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(dec_channel[4], dec_channel[4], kernel_size=3, stride=1, padding=1, residual=True), ),
        #     # 24, 24
        #
        #     nn.Sequential(Conv2dTranspose(upsamp_channel[4], dec_channel[5], kernel_size=3, stride=2, padding=1,
        #                                   output_padding=1),
        #                   Conv2d(dec_channel[5], dec_channel[5], kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(dec_channel[5], dec_channel[5], kernel_size=3, stride=1, padding=1, residual=True), ),
        #     # 48, 48
        #
        #     nn.Sequential(Conv2dTranspose(upsamp_channel[5], dec_channel[6], kernel_size=3, stride=2, padding=1,
        #                                   output_padding=1),
        #                   Conv2d(dec_channel[6], dec_channel[6], kernel_size=3, stride=1, padding=1, residual=True),
        #                   Conv2d(dec_channel[6], dec_channel[6], kernel_size=3, stride=1, padding=1,
        #                          residual=True), ), ])  # 96,96

        # self.emotion_decoder = nn.Sequential(
        #
        # )

        # self.output_block = nn.Sequential(Conv2d(upsamp_channel[6], 32, kernel_size=3, stride=1, padding=1),
        #                                   nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #                                   nn.Sigmoid())
        self.ft = False
        self.sync_net = av_sync(audio_num, audio_num)

    def forward(self, sample, idAudio):
        # input 1*F*T
        with torch.no_grad() if not self.ft else contextlib.ExitStack():
            enc_out = self.audio_encoder(**sample["net_input"])
        # T*B*C, B*T

        audio_embedding, audio_padding = enc_out['encoder_out'], enc_out['padding_mask']
        # emtion_in = sample['emotion'].unsqueeze(1).repeat(1, 15, 1)
        x = audio_embedding.permute(1, 0, 2).reshape(-1, audio_embedding.shape[2])[idAudio]
        x = self.audio_map(x)
        x = x.reshape(x.shape[0], 512, 1, 1)

        return x, audio_embedding

        # return outputs, audio_embedding  # , ys_hat#, wer

    def get_aud_emb(self, sample):
        with torch.no_grad():
            enc_out = self.audio_encoder(**sample["net_input"])
        # T*B*C, B*T

        audio_embedding, audio_padding = enc_out['encoder_out'], enc_out['padding_mask']
        return audio_embedding


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        enc_channel = [6, 16, 32, 64, 128, 256, 512, 512]
        dec_channel = [512, 512, 512, 384, 256, 128, 64]
        upsamp_channel = []
        res_layers = None
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
            nn.Sequential(
                Conv2d(1024, dec_channel[0], kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1536, dec_channel[1], kernel_size=3, stride=1, padding=0),  # 3,3
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

        self.output_block = nn.Sequential(Conv2d(upsamp_channel[6], 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, x, feats, emotion_embedding, input_dim_size, B):
        feats_t = feats
        x = torch.cat((x, emotion_embedding), dim=1)

        emotion_size = emotion_embedding.size(0)
        feats_size = feats_t[-1].size(0)

        if emotion_size < feats_size:
            padding_size = feats_size - emotion_size
            padding = (0, 0, 0, 0, 0, 0, 0, padding_size)
            emotion_embedding = F.pad(emotion_embedding, padding)

        elif feats_size < emotion_size:
            padding_size = emotion_size - feats_size
            padding = (0, 0, 0, 0, 0, 0, 0, padding_size)
            for i in range(len(feats_t)):
                feats_t[i] = F.pad(feats_t[i], padding)

        feats_t[-1] = torch.cat([emotion_embedding, feats_t[-1]], dim=1)

        for i, f in enumerate(self.face_decoder_blocks):
            if i == 0:
                x1 = f(x)
            else:
                x1 = f(x1)
            try:
                if i < self.res_layers:
                    feat = feats_t[-1]
                    if x1.shape[0] != feat.shape[0]:
                        # Determine the padding size
                        diff = abs(x1.shape[0] - feat.shape[0])
                        if x1.shape[0] < feat.shape[0]:
                            x1 = F.pad(x1, (0, 0, 0, 0, 0, 0, 0, diff))
                        else:
                            feat = F.pad(feat, (0, 0, 0, 0, 0, 0, 0, diff))
                    x1 = torch.cat((x1, feat), dim=1)
            except Exception as e:
                print(x1.size())
                raise e

            feats_t.pop()

        x1 = self.output_block(x1)
        # outputs = x1

        if input_dim_size > 4:
            x1 = torch.split(x1, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x1, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x1
        print('OUTPUT**********************' + str(outputs.shape))

        return outputs


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
    def __init__(self, debug=False):
        super(DISCEMO, self).__init__()
        # self.args = args
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
            nn.Linear(4608, 2048),
            nn.LeakyReLU(0.3),
            nn.Linear(2048, 512)
        )

        self.rnn_1 = nn.LSTM(512, 512, 1, bidirectional=False, batch_first=True)

        self.cls = nn.Sequential(
            nn.Linear(512, 6)
        )

        # Optimizer
        # self.opt = optim.Adam(list(self.parameters()), lr=self.args.lr_emo, betas=(0.5, 0.999))
        # self.opt = optim.RMSprop(list(self.parameters()), lr = params['LR_DE'])
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)
        self.lossE = nn.CrossEntropyLoss()
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.95, last_epoch=-1)

    def percep_forward(self, video, r_emotion):
        x = video
        t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.contiguous().view(t, c, w, h)

        for i in range(len(self.filters)):
            x = getattr(self, 'conv_' + str(i + 1))(x)
        h = x.view(t, -1)
        h = self.projector(h)

        h, _ = self.rnn_1(h)

        h_class = self.cls(h[:, -1, :])

        lossE = self.lossE(h_class[0], r_emotion[0])
        return lossE

    # def forward(self, video):
    #     x = video
    #     n, c, t, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
    #     x = x.contiguous().view(t * n, c, w, h)
    #
    #     for i in range(len(self.filters)):
    #         x = getattr(self, 'conv_' + str(i + 1))(x)
    #     h = x.view(n, t, -1)
    #     h = self.projector(h)
    #
    #     h, _ = self.rnn_1(h)
    #
    #     h_class = self.cls(h[:, -1, :])
    #
    #     return h_class

    def forward(self, video):
        x = video
        t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.contiguous().view(t, c, w, h)

        for i in range(len(self.filters)):
            x = getattr(self, 'conv_' + str(i + 1))(x)
        h = x.view(t, -1)
        h = self.projector(h)

        h, _ = self.rnn_1(h)

        h_class = self.cls(h[-1, :])
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
