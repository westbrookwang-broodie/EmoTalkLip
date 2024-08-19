import argparse
import json
import os
from tqdm import tqdm
import random as rn
import shutil
from os.path import dirname, join, basename

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from models.talklip import DISCEMO
from datagen_aug import Dataset


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-path", type=str, help="Input folder containing train data", default=None,
                        required=True)
    # parser.add_argument("-v", "--val-path", type=str, help="Input folder containing validation data", default=None, required=True)
    parser.add_argument("-o", "--out_path", type=str, help="output folder", default='../models/def', required=True)

    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument('--lr_emo', type=float, default=1e-06)

    parser.add_argument("--gpu-no", type=str, help="select gpu", default='2')
    parser.add_argument('--seed', type=int, default=9)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    args.batch_size = args.batch_size * max(int(torch.cuda.device_count()), 1)
    args.steplr = 200

    args.filters = [64, 128, 256, 512, 512]
    # -----------------------------------------#
    #           Reproducible results          #
    # -----------------------------------------#
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)
    torch.manual_seed(args.seed)
    # -----------------------------------------#

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    else:
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)

    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    return args


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def enableGrad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def train():
    args = initParams()

    trainDset = Dataset(args)

    train_loader = torch.utils.data.DataLoader(trainDset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)

    disc_emo = DISCEMO().to(args.device)
    disc_emo.apply(init_weights)
    dis_emo_optimizer = torch.optim.Adam([p for p in disc_emo.parameters() if p.requires_grad], lr=1e-4, betas=(0.5, 0.999))
    dis_emo_sch = torch.optim.lr_scheduler.StepLR(dis_emo_optimizer, 150, gamma=0.1, last_epoch=-1)
    # disc_emo = nn.DataParallel(disc_emo, device_ids)

    emo_loss_disc = nn.CrossEntropyLoss()

    num_batches = len(train_loader)
    print(args.batch_size, num_batches)

    global_step = 0

    for epoch in range(args.num_epochs):
        print('Epoch: {}'.format(epoch))
        prog_bar = tqdm(enumerate(train_loader))
        running_loss = 0.
        for step, (x, y) in prog_bar:
            video, emotion = x.to(args.device), y.to(args.device)

            disc_emo.train()

            dis_emo_optimizer.zero_grad()  # .module is because of nn.DataParallel

            print('VIDEO')
            print(video.shape)

            class_real = disc_emo(video)

            loss = emo_loss_disc(class_real, torch.argmax(emotion, dim=1))

            running_loss += loss.item()

            loss.backward()
            dis_emo_optimizer.step()  # .module is because of nn.DataParallel

            if global_step % 1000 == 0:
                print('Saving the network')
                save_checkpoint(disc_emo, dis_emo_optimizer, dis_emo_sch, global_step, args.out_path, epoch, 'emo_disc')
                print('Network has been saved')
                print('step is' + str(global_step))

            prog_bar.set_description('classification Loss: {}'.format(running_loss / (step + 1)))

            global_step += 1


        dis_emo_sch.step()  # .module is because of nn.DataParallel


def save_checkpoint(model, optimizer,scheduler, global_step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    train()
