import os

dir_path = '/home/star/lipread_mp4'
file_ls = os.listdir(dir_path)

for file in file_ls:
    if file.startswith('A'):
        path_min = os.path.join(dir_path, file)
        file_exts = os.listdir(path_min + '/test')
        for file_ext in file_exts:
            if file_ext.endswith('.mp4'):
                print(path_min + '/test' + '/' + file_ext.replace('.mp4', ''))
# import cv2
#
#
# def load_video_spec(sample):
#     cap = cv2.VideoCapture('{}.mp4'.format(sample))
#     txt = '{}.txt'.format(sample)
#     with open(txt, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.find('Duration') != -1:
#                 duration = float(line.split(' ')[1])
#
#     fps = cap.get(5)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     video_length = frame_count / fps
#     mid_frame = frame_count // 2
#     start_frame = max(mid_frame - int(duration * fps), 0)
#     end_frame = min(mid_frame + int(duration * fps), frame_count)
#     imgs = []
#     out = cv2.VideoWriter('out1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#     # 跳转到开始帧
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
#     # 读取帧并写入新的视频文件
#     current_frame = start_frame
#     while current_frame < end_frame:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         else:
#             imgs.append(frame)
#         current_frame += 1
#         out.write(frame)
#
#     # 释放所有资源
#     cap.release()
#     out.release()
#     return imgs


# load_video_spec('/home/star/lipread_mp4/AMERICA/train/AMERICA_00119')