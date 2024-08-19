# EmoTalkLip

<img src="https://media.giphy.com/media/95NxuBd07zpciqwkt6/giphy.gif" style="zoom:150%;" /><img src="https://media.giphy.com/media/hbPb4BX8hSYZAC1eZu/giphy.gif" style="zoom:150%;" />

<img src="https://media.giphy.com/media/qNYmA5iKGYVeI54cab/giphy.gif" style="zoom: 200%;" /><img src="https://media.giphy.com/media/u9FmVng5P1TvMbKzD8/giphy.gif" style="zoom: 200%;" /><img src="https://media.giphy.com/media/FQ7Frv6JBbCCazIjYA/giphy.gif" style="zoom: 200%;" /><img src="https://media.giphy.com/media/W0FKf6SlwgyvuT11Mw/giphy.gif" style="zoom: 200%;" />
## Prerequisite

1. `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html` .
2. Install [AV-hubert](https://github.com/facebookresearch/av_hubert) by following his installation
3. Install supplementary packages via `pip install -r requirements.txt`
4. Install ffmpeg.
5. Install [DiffBIR](https://github.com/XPixelGroup/DiffBIR) by following his installation.  Please be attention that we use the version 1.13.1+cu116 so please install xformers 0.0.16.
6. Download the pre-trained checkpoint of face detector [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) and put it to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8).

## Dataset and pre-processing

1. Download the [LR2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) for training and evaluation.
2. Download the [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) for evaluation.
3. Download the Crema-d from the [repo](https://github.com/CheyneyComputerScience/CREMA-D).  According to the repo that repo's preparing **crema-d for training** subtitle to **convert videos to 25 fps** and **preprocess dataset**.
4. To extract wavforms from mp4 files:

```python
python preparation/audio_extract.py --filelist $filelist  --video_root $video_root --audio_root $audio_root
```

- $filelist: a txt file containing names of videos. We provide the filelist of LRW test set as an example in the datalist directory.
- $video_root: root directory of videos. In LRS2 dataset, $video_root should contains directories like "639XXX". In LRW dataset, $video_root should contains directories like "ABOUT".
- $audio_root: root directory for saving wavforms
- other optional arguments: please refer to audio_extract.py

5. To extract wavforms from mp4 files:

```python
python preparation/bbx_extract.py --filelist $filelist  --video_root $video_root --bbx_root $bbx_root --gpu $gpu
```

- $bbx_root: a root directory for saving detected bounding boxes
- $gpu: run bbx_extract on a specific gpu. For example, 3.

## Train!

After installing the AV-Hubert, some files need to be modified.

```python
rm xxx/av_hubert/avhubert/hubert_asr.py
cp avhubert_modification/hubert_asr_wav2lip.py xxx/av_hubert/avhubert/hubert_asr.py

rm xxx/av_hubert/fairseq/fairseq/criterions/label_smoothed_cross_entropy.py
cp avhubert_modification/label_smoothed_cross_entropy_wav2lip.py xxx/av_hubert/fairseq/fairseq/criterions/label_smoothed_cross_entropy.py
```



We provide the pre-trained emotion-discrinminator. You can either use ours into $emo_disc_checkpoint_path or train your owns through [emogen](https://github.com/sahilg06/EmoGen). 

You can train with the following command.

```python
python train_in_turn.py --file_dir $file_dir_for_crema_d --file_dir_2 $file_name_for_lrs2 --video_root $video_root_for_crema_d --audio_root $audio_root_for_crema_d --bbx_root $bbx_root_for_crema_d --word_root $word_root_for_lrs2 --avhubert_root $avhubert_root --avhubert_path $avhubert_path --checkpoint_dir $checkpoint_dir --gpu $num --gen_checkpoint_path $gen_checkpoint_path --disc_checkpoint_path $disc_checkpoint_path --emoGen_checkpoint_path $emogen_checkpoint_path --decoder_checkpoint_path $decoder_checkpoint_path --emo_disc_checkpoint_path $emo_disc_checkpoint_path 
```

- $file_dir: a directory which contains filename.txt of Crema-d dataset
- $file_dir_2: a directory which contains train.txt, valid.txt, test.txt of LRS2 dataset
- $word_root: root directory of text annotation. Normally, it should be equal to $video_root, as LRS2 dataset puts a video file ".mp4" and its corresponding text file ".txt" in the same directory.
- $avhubert_root: path of root of avhubert (should like xxx/av_hubert)
- $avhubert_path: download the above Lip reading expert and enter its path
- $checkpoint_dir: a directory to save checkpoint of talklip
- $gen_checkpoint_dir(optional): enter the path of a generator checkpoint if you want to resume training from a checkpoint
- $disc_checkpoint_dir(optional): enter the path of a discriminator checkpoint if you want to resume training from a checkpoint
- $emoGen_checkpoint_dir(optional): enter the path of an emotion encoder checkpoint if you want to resume training from a checkpoint
- $decoder_checkpoint_dir(optional): enter the path of a decider checkpoint if you want to resume training from a checkpoint
- $emo_disc_checkpoint_dir(optional): enter the path of an emotion discriminator checkpoint if you want to resume training from a checkpoint
- $log_name: name of log file
- $cont_w: weight of contrastive learning loss (default: 1e-3)
- $lip_w: weight of lip reading loss (default: 1e-5)
- $perp_w: weight of perceptual loss (default: 0.07)
- $e_w: weight of emotion loss (default: 0.01)



## Test

To test the model without emotion, you can follow the below command:

```python
python inf_test.py --filelist $filelist --video_root $video_root --audio_root $audio_root \
--bbx_root $bbx_root --save_root $syn_video_root --ckpt_path $generator_ckpt --ckpt_emo_path $emotion_ckpt --ckpt_decoder_path $decoder_ckpt --avhubert_root $avhubert_root
```

- $filelist: a txt file containing names of all test files
- $syn_video_root: root directory for saving synthesized videos
- $generator_ckpt, emotion_ckpt, decoder_ckpt: a trained checkpoint of EmoTalkLip net

To test the model with all the emotions, you can follow the below command:

```
python inf_all_emotions.py --filelist $filelist --video_root $video_root --audio_root $audio_root \
--bbx_root $bbx_root --save_root $syn_video_root --ckpt_path $generator_ckpt --ckpt_emo_path $emotion_ckpt --ckpt_decoder_path $decoder_ckpt --avhubert_root $avhubert_root
```



## Inference_demo

You can inference a demo by the below code:

```python
python inf_demo.py --video_path $video_path --wav_path $wav_path --avhubert_root $avhubert_root --emotion $emotion --ckpt_path $generator_ckpt --emotion_encoder_ckpt_path $emotion_ckpt --decoder_ckpt_path $decoder_ckpt --save_path $syn_video_root --version $ver --task $ta --upscale $us  --cfg_scale $cfgs --output $ouput_path
```

- $video_file: a video file (end with .mp4)
- $audio_file: an audio file (end with .wav)
- $emotion: an emotion label in [ANG, DIS, FEA, HAP, NEU, SAD]
- $ver, $ta, $us, $cfgs, $output, you can simply look up in **Blind Face Restoration** in [DiffBIR](https://github.com/XPixelGroup/DiffBIR)



## Evaluation

Please follow README.md in the evaluation directory