# Author: AP-Kai
# Datetime: 2022/1/10
# Copyright belongs to the author.
# Please indicate the source for reprinting.


class Member:
    def __init__(self):
        self.video_path = r'D:\Python_Projects\EIVideo\EIVideo\example\example.mp4'
        # self.one_json_path = 'E:/PaddlePaddle_Project/EIVideo/resources/test_data.json'  # 单个帧
        self.one_json_path = r'D:\Python_Projects\EIVideo\resources/test_data.json'
        self.submit_masks_json_path = r'D:\Python_Projects\EIVideo\resources/masks.json'  # 最终结果
        # E:/PaddlePaddle_Project/EIVideo/EIVideo/masks.json
        self.inter_file_path = r'D:\Python_Projects\EIVideo\resources'
        self.weights_path = r'D:\Python_Projects\EIVideo/save_step_80000.pdparams'


m = Member()
