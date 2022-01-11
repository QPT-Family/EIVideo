# Author: Acer Zhang
# Datetime: 2022/1/6 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os

EI_VIDEO_ROOT = os.path.abspath(os.path.dirname(__file__))
TEMP_IMG_SAVE_PATH = "./temp.png"
TEMP_JSON_SAVE_PATH = "Users/liuchen21/Library/Mobile Documents/com~apple~CloudDocs/Documents/001.json"
TEMP_JSON_FINAL_PATH = "./final.json"


def join_root_path(path: str):
    return os.path.join(EI_VIDEO_ROOT, path)
