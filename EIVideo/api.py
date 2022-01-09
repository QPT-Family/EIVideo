import json
import os

import cv2
import numpy as np
from PIL import Image

from resources.backend.paddlevideo.utils.manet_utils import overlay_davis


def load_video(path, min_side=None):
    frame_list = []
    cap = cv2.VideoCapture(path)
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
    return frames


def get_scribbles():
    for i in range(8):
        with open(f'/home/lc/paddlevideo/data/bike-packing/lable/{i + 1}.json'
                  ) as f:
            scribbles = json.load(f)
            first_scribble = not i
            yield scribbles, first_scribble


def get_images(sequence='bike-packing'):
    img_path = os.path.join('data', sequence.strip(), 'frame')
    img_files = os.listdir(img_path)
    img_files.sort()
    files = []
    for img in img_files:
        img_file = np.array(Image.open(os.path.join(img_path, img)))
        files.append(img_file)
    return np.array(files)
def submit_masks(masks, images, inter_file_path):
    overlays = []
    save_result_path = os.path.join(inter_file_path, 'result')
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
    with open(os.path.join(save_result_path, 'masks.json'), 'w') as f:
        json.dump(result, f)
