# Author: AP-Kai
# Datetime: 2022/1/10
# Copyright belongs to the author.
# Please indicate the source for reprinting.


class member():
    def __init__(self):
        self.video_path = 'E:/PaddlePaddle_Project/EIVideo/EIVideo/example/example1.mp4'
        self.one_json_path = 'E:/PaddlePaddle_Project/EIVideo/resources/test_data.json'  # 单个帧
        self.submit_masks_json_path = 'E:/PaddlePaddle_Project/EIVideo/resources/masks.json'  # 最终结果
        # E:/PaddlePaddle_Project/EIVideo/EIVideo/masks.json
        self.image_path = 'tmp.png'
        self.inter_file_path = 'E:/PaddlePaddle_Project/EIVideo/resources/'
        self.weights_path = 'E:/PaddlePaddle_Project/EIVideo/EIVideo/model/save_step_80000.pdparams'


m = member()
