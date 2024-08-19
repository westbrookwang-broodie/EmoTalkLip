import cv2
import os

# 视频文件路径
video_dir = '/home/star/mvlrs_v1/main'
file_name = '/home/star/wy/test.txt'

# 保存帧的文件夹路径
frame_folder = 'opt_hap_3'

video_path = "/home/star/wy/RESFFFFF_2.mp4"
name = ""

# 如果保存帧的文件夹不存在，则创建它
if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)

cap = cv2.VideoCapture(video_path)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print('------')
        break

    # 保存帧为图片
    frame_name = f'{name}_{frame_count:04d}.jpg'
    print(frame_name)
    frame_path = os.path.join(frame_folder, frame_name)
    print(frame_path)
    cv2.imwrite(frame_path, frame)

    frame_count += 1

# 关闭视频文件
cap.release()

# with open(file_name, 'r') as f:
#     lines = f.readlines()
#
# for line in lines:
#     l1 = line.split(" ")[0]
#     if not os.path.exists(os.path.join(frame_folder, l1)):
#         os.makedirs(os.path.join(frame_folder, l1))
#     dir_name = os.path.join(frame_folder, l1)
#     path = os.path.join(video_dir, l1) + '.mp4'
#     name = l1.split('/')[-1]
#     cap = cv2.VideoCapture(path)
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if not ret:
#             print('------')
#             break
#
#         # 保存帧为图片
#         frame_name = f'{name}_{frame_count:04d}.jpg'
#         print(frame_name)
#         frame_path = os.path.join(dir_name, frame_name)
#         print(frame_path)
#         cv2.imwrite(frame_path, frame)
#
#         frame_count += 1
#
#     # 关闭视频文件
#     cap.release()

# for file in os.listdir(video_dir):
#     if file.endswith('.mp4'):
#         name = file.split('.')[0]
#         cap = cv2.VideoCapture(os.path.join(video_dir, file))
#         # os.makedirs(os.path.join(frame_folder, name))
#
#         # 读取视频帧并保存为图片
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#
#             if not ret:
#                 print('------')
#                 break
#
#             # 保存帧为图片
#             frame_name = f'{name}_{frame_count:04d}.jpg'
#             print(frame_name)
#             frame_path = os.path.join(frame_folder, frame_name)
#             cv2.imwrite(frame_path, frame)
#
#             frame_count += 1
#
#         # 关闭视频文件
#         cap.release()
#
#
# # 打开视频文件

