#!/bin/bash
conda init bash
cd /ssd2/m3lab/usrs/wy/TalkLip
source activate avhubert
conda env list
/home/wangjinbao/bin/python inf_demo.py --video_path /ssd2/m3lab/usrs/wy/HAP.mp4 --wav_path /ssd2/m3lab/usrs/wy/test.wav --ckpt_path /ssd2/m3lab/usrs/wy/TalkLip/real_check/checkpoint_step000106000.pth --avhubert_root /ssd2/m3lab/usrs/wy/av_hubert --emotion HAP