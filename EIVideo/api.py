# Author: AP-Kai
# Datetime: 2022/1/10
# Copyright belongs to the author.
# Please indicate the source for reprinting.


import json
import os
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image

from resources.backend.paddlevideo.utils.manet_utils import overlay_davis

from EIVideo.member import m

# use this
def load_video(min_side=None):
    print("now loading video")
    print('this is video_path  ' + m.video_path)
    frame_list = []
    cap = cv2.VideoCapture(m.video_path)
    while (cap.isOpened()):
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


def get_images(sequence='bike-packing'):
    img_path = os.path.join('data', sequence.strip(), 'frame')
    img_files = os.listdir(img_path)
    img_files.sort()
    files = []
    for img in img_files:
        img_file = np.array(Image.open(os.path.join(img_path, img)))
        files.append(img_file)
    return np.array(files)


def get_scribbles():
    for i in range(8):
        with open(m.json_path) as f:
            scribbles = json.load(f)
            first_scribble = not i
            yield scribbles, first_scribble


def submit_masks(masks, images):
    overlays = []
    save_result_path = os.path.join(m.inter_file_path, 'result')
    os.makedirs(save_result_path, exist_ok=True)
    for imgname, (mask, image) in enumerate(zip(masks, images)):
        overlay = overlay_davis(image, mask)
        overlays.append(overlay.tolist())
        overlay = Image.fromarray(overlay)
        imgname = str(imgname)
        while len(imgname) < 5:
            imgname = '0' + imgname
        overlay.save(os.path.join(save_result_path, imgname + '.png'))
    result = {'overlays': overlays}
    # result = {'masks': masks.tolist()}
    m.submit_masks_json_path = os.path.join(save_result_path, "masks.json")
    with open(m.submit_masks_json_path, 'w') as f:
        json.dump(result, f)


def json2frame(path):
    print(path)
    with open(path, 'r', encoding='utf-8') as f:
        res = f.read()
        a = json.loads(res)
        b = a.get('overlays')
        b_array = np.array(b)
        frame_list = []

        for i in range(0, len(b_array)):
            im = Image.fromarray(np.uint8(b_array[i]))
            im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            frame_list.append(im)
    return frame_list


def png2json(image_path, sliderframenum, save_json_path):
    image = Image.open(image_path)  # 用PIL中的Image.open打开图像
    image = image.convert('P')
    image_arr = np.array(image)  # 转化成numpy数组
    image_arr = image_arr.astype("float32")
    r1 = np.argwhere(image_arr == 1)  # tuple
    pframes = []
    # i -> object id
    for i in range(1, len(np.unique(image_arr))):
        pframe = OrderedDict()
        pframe['path'] = []
        # Find object id in image_arr
        r1 = np.argwhere(image_arr == i)  # tuple
        r1 = r1.astype("float32")
        # Add path to pframe
        for j in range(0, len(r1)):
            r1[j][0] = r1[j][0] / 480.0
            r1[j][1] = r1[j][1] / 910.0
            # r1[j] = np.around(r1[j], decimals=16)
            pframe['path'].append(r1[j].tolist())
        # Add object id, start_time, stop_time
        pframe['object_id'] = i
        pframe['start_time'] = sliderframenum
        pframe['stop_time'] = sliderframenum
        # Add pframe to pframes
        pframes.append(pframe)

    dic = OrderedDict()
    dic['scribbles'] = []
    for i in range(0, int(100)):
        if i == sliderframenum:
            # Add value to frame[]
            dic['scribbles'].append(pframes)
        else:
            dic['scribbles'].append([])

    json_str = json.dumps(dic)
    with open(save_json_path, 'w') as json_file:
        json_file.write(json_str)
