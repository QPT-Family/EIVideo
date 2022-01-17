# Author: AP-Kai
# Datetime: 2022/1/10
# Copyright belongs to the author.
# Please indicate the source for reprinting.


import json
import os
import time
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image

from EIVideo.paddlevideo.utils.manet_utils import overlay_davis
from EIVideo import TEMP_JSON_SAVE_PATH, TEMP_JSON_FINAL_PATH


def get_images(sequence='bike-packing'):
    img_path = os.path.join('data', sequence.strip(), 'frame')
    img_files = os.listdir(img_path)
    img_files.sort()
    files = []
    for img in img_files:
        img_file = np.array(Image.open(os.path.join(img_path, img)))
        files.append(img_file)
    return np.array(files)


def json2frame(overlays):
    arr = np.array(json.loads(overlays)['overlays']).astype('uint8')
    frame_list = []
    for i in range(0, len(arr)):
        im = Image.fromarray(np.uint8(arr[i]))
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        frame_list.append(im)
    return frame_list


def png2json(image_path, sliderframenum=0, first_scribble=False):
    image_ = Image.open(image_path)  # 用PIL中的Image.open打开图像
    image = image_.convert('P')
    image_arr = np.array(image)  # 转化成numpy数组
    image_arr = image_arr.astype("float32")
    pframes = []
    # i -> object id
    for i in range(1, len(np.unique(image_arr))):
        pframe = OrderedDict()
        pframe['path'] = []
        # Find object id in image_arr
        r1 = np.argwhere(image_arr == i)  # tuple
        r1 = r1.astype("float32")
        # Add path to pframe
        r1 /= image_arr.shape
        r1[:, [0, 1]] = r1[:, [1, 0]]
        pframe['path'] = r1.tolist()
        # Add object id, start_time, stop_time
        pframe['object_id'] = i
        pframe['start_time'] = sliderframenum
        pframe['stop_time'] = sliderframenum
        # Add pframe to pframes
        pframes.append(pframe)

    dic = OrderedDict()
    dic['first_scribble'] = first_scribble
    dic['scribbles'] = []
    for i in range(0, int(150)):
        if i == sliderframenum:
            # Add value to frame[]
            dic['scribbles'].append(pframes)
        else:
            dic['scribbles'].append([])
    json_str = json.dumps(dic)
    with open('save.json', 'w') as f:
        f.write(json_str)
    return json_str


def load_video(video_path, min_side=None):
    frame_list = []
    # ToDo To AP-kai: 是不是轻松干掉了m.video_path？
    cap = cv2.VideoCapture(video_path)
    # ToDo To AP-kai: while (cap.isOpened()): -> 不必多写个括号哈
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if min_side:
            h, w = frame.shape[:2]
            new_w = (w * min_side // min(w, h))
            new_h = (h * min_side // min(w, h))
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
            # .transpose([2, 0, 1])
        frame_list.append(frame)
    frames = np.stack(frame_list, axis=0)
    return frames, frame_list


def get_scribbles(client_socket):
    from_client_msg = recv_end(client_socket)
    from_client_msg = json.loads(from_client_msg)
    scribbles, first_scribble = from_client_msg, from_client_msg['first_scribble']
    return scribbles, first_scribble


End = '$'


def recv_end(the_socket):
    total_data = []
    while True:
        data = the_socket.recv(8192).decode("gbk")
        print(data)
        if End in data:
            total_data.append(data[:data.find(End)])
            break
        total_data.append(data)
        if len(total_data) > 1:
            # check if end_of_data was split
            last_pair = total_data[-2] + total_data[-1]
            if End in last_pair:
                total_data[-2] = last_pair[:last_pair.find(End)]
                total_data.pop()
                break
    return ''.join(total_data)


def get_overlays(masks, images, save_path=None):
    overlays = []
    masks = np.array(masks).astype('uint8')
    for img_name, (mask, image) in enumerate(zip(masks, images)):
        overlay = overlay_davis(image, mask)
        overlays.append(overlay.tolist())
        if save_path is not None:
            overlay = Image.fromarray(overlay)
            img_name = str(img_name)
            while len(img_name) < 5:
                img_name = '0' + img_name
            overlay.save(os.path.join(save_path, img_name + '.png'))
    return overlays


def submit_masks(images, masks, the_socket):
    overlays = get_overlays(masks, images)
    result = json.dumps({'overlays': overlays})
    the_socket.send(bytes(result + '$', encoding="gbk"))
    # with open(TEMP_JSON_FINAL_PATH, 'w') as f:
    #     json.dump(result, f)