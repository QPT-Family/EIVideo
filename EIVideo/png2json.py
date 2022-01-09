from PIL import Image
import numpy as np
import pandas as pd
import json
from collections import OrderedDict

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


if __name__ == '__main__':
    png2json("123.png", 10, 'test_data.json')


