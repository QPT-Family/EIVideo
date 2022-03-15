# Author: Acer Zhang
# Datetime: 2022/1/6 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os
from EIVideo.version import __version__
from EIVideo.eivideo.utils.log import Logging

EI_VIDEO_ROOT = os.path.abspath(os.path.dirname(__file__))
TEMP_IMG_SAVE_PATH = "./QEIVideo/temp.png"
TEMP_JSON_FINAL_PATH = "./EIVideo/final.json"
CONFIG_FILE_PATH = "./EIVideo/configs/manet.cfg"


def join_root_path(path: str):
    return os.path.join(EI_VIDEO_ROOT, path)


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        Logging.debug('delete temp file(s) done')
    else:
        Logging.debug('no such file:%s' % file_path)
