import json
from collections import OrderedDict

import numpy
import numpy as np
from PIL import Image
from davisinteractive.utils.scribbles import scribbles2mask


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


from EIVideo.paddlevideo.utils.manet_utils import _palette


def resave_img(num=1, save_path='resave.png', json_path='/Users/liuchen21/Library/Mobile Documents/com~apple~CloudDocs/Documents/PycharmProjects/EIVideo/QEIVideo/save.json'):
    with open(json_path, 'r') as f:
        scribbles = json.load(f)
    scribble_masks = scribbles2mask(scribbles, (500, 500), bresenham=False)
    scribble_label = scribble_masks[num]
    ref_scribble_to_show = numpy.array(scribble_label)
    print(len(np.where(ref_scribble_to_show == 1)[0]))
    print(np.array(scribbles['scribbles'][num][0]['path']).shape)
    im_ = Image.fromarray(
        ref_scribble_to_show.astype('uint8')).convert('P', )
    im_.putpalette(_palette)
    im_.save(save_path)


# png2json(
#     '/Users/liuchen21/Library/Mobile Documents/com~apple~CloudDocs/Documents/PycharmProjects/EIVideo/output/Manet/swan/interactive1/turn1/inter_25.png')
resave_img()
# resave_img(save_path='right.png', json_path='/Users/liuchen21/Desktop/001.json', num=25)
