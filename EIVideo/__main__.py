# Author: AP-Kai
# Datetime: 2022/3/15
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import random
import os
import json
import paddle
import numpy as np

from flask import Flask, jsonify, request
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import EIVideo
from EIVideo import join_root_path, delete_file, CONFIG_FILE_PATH
from EIVideo.eivideo.core import Manet
from EIVideo.eivideo.utils.manet import get_dist_info
from EIVideo.eivideo.utils import get_config, Logging

DEF_CONFIG_FILE_PATH = join_root_path("configs/manet.yaml")
DEF_PARAMS_FILE_PATH = join_root_path("model/default_manet.pdparams")

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'data/uploads/video')
ALLOWED_EXTENSIONS = set(["mp4", "MP4", "mov", "MOV", "avi", "AVI"])
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

EIVideo_ROOT = os.path.dirname(EIVideo.__file__)
MODEL_PATH = os.path.join(EIVideo_ROOT, "model/default_manet.pdparams")
if not os.path.exists(MODEL_PATH):
    import wget
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    Logging.info("正在下载模型文件")
    wget.download("https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MANet_EIVideo.pdparams", out=MODEL_PATH)
    Logging.info("模型文件下载完成")


def start_infer(video_path=None, save_path=None, json_scribbles=None):
    paddle.set_device("gpu")
    cfg = get_config(CONFIG_FILE_PATH)
    Logging.info(str(cfg))
    # set seed if specified
    seed = None
    seed = seed
    if seed is not None:
        assert isinstance(
            seed,
            int), f"seed must be a integer when specified, but got {seed}"
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    cfg_helper = {"knns": 1, "is_save_image": True}
    cfg.update(cfg_helper)
    Logging.debug(str(cfg))

    Logging.info("开始预测")
    Manet().test_step(**cfg, save_path=save_path, video_path=video_path, weights=DEF_PARAMS_FILE_PATH,
                      parallel=parallel, json_scribbles=json_scribbles
                      )


@app.route('/', methods=['GET'])
def hello():
    return "hello world"


@app.route('/infer', methods=['GET', 'POST'])
def infer():
    if request.method == 'POST':
        data = request.get_data()
        json_data = json.loads(data.decode("utf-8"))
        video_name = json_data.get("video_path").split('/')[-1]
        save_path = json_data.get("save_path")
        json_scribbles = json_data.get("params")
        start_infer(video_path=video_name, save_path=save_path, json_scribbles=json_scribbles)
        from EIVideo import TEMP_JSON_FINAL_PATH
        with open(TEMP_JSON_FINAL_PATH) as f:
            jsonStr = json.load(f)
            delete_file(TEMP_JSON_FINAL_PATH)
            return jsonify(jsonStr)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return "It's empty"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({
                "isUploaded": "Uploaded done",
                "filename": filename
            })
        return jsonify({
            "isUploaded": "Upload failed",
            "filename": None
        })


if __name__ == '__main__':
    paddle.set_device("gpu")
    Logging.info("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    Logging.info("-+-+-+-+-+服务启动成功-+-+-+-+-")
    Logging.info("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    app.run(debug=False, host='0.0.0.0', port=6666)
