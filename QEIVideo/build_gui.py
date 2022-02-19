# Author: AP-Kai
# Datetime:2022/2/17
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import json
import os
import requests
import cv2

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import EIVideo
from EIVideo.api import json2frame, png2dic, load_video
from EIVideo import TEMP_IMG_SAVE_PATH, TEMP_JSON_FINAL_PATH
from QEIVideo.log import Logging
from QEIVideo.gui.ui_main_window import Ui_MainWindow

EIVideo_ROOT = os.path.dirname(EIVideo.__file__)
MODEL_PATH = os.path.join(EIVideo_ROOT, "model/default_manet.pdparams")
if not os.path.exists(MODEL_PATH):
    import wget
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    Logging.info("正在下载模型文件")
    wget.download("https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MANet_EIVideo.pdparams", out=MODEL_PATH)
    Logging.info("模型文件下载完成")


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        Logging.debug('delete temp file(s) done')
    else:
        Logging.debug('no such file:%s' % file_path)  # 则返回文件不存在


class BuildGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(BuildGUI, self).__init__()
        self.select_video_path = None
        self.save_path = "./result"
        os.makedirs(self.save_path, exist_ok=True)

        self.setupUi(self)
        Logging.info("QEIVideo启动成功")
        QMessageBox.information(self,
                                "使用必读",
                                "当前程式为示例程式，仅供模型效果演示，且未对操作系统、硬件进行强制限制，"
                                "可能会因环境而造成意料之外/无法正确使用的情况。\n"
                                "目前我们正在设计新版本EIVideo来提供更好的使用体验，也欢迎通过GitHub issue的形式联系并加入我们~\n"
                                "https://github.com/QPT-Family/EIVideo",
                                QMessageBox.Yes)

    def infer(self):
        self.label.setText("Start infer")
        self.progressBar.setProperty("value", 0)
        image = self.paintBoard.get_content_as_q_image()
        image.save(TEMP_IMG_SAVE_PATH)
        Logging.debug("infer frame num: "+str(self.slider_frame_num))
        self.progressBar.setProperty("value", 25)
        dic_str = png2dic(TEMP_IMG_SAVE_PATH, self.slider_frame_num)
        self.progressBar.setProperty("value", 50)
        paths2 = {"video_path": self.select_video_path,
                  "save_path": self.save_path,
                  "params": dic_str}
        paths_json = json.dumps(paths2)
        r = requests.post("http://127.0.0.1:5000/infer", data=paths_json)

        Logging.info("推理结束,正在拉取结果.")
        self.progressBar.setProperty("value", 75)
        self.all_frames = json2frame(path=TEMP_JSON_FINAL_PATH)
        Logging.info("拉取结果成功")
        self.update_frame()
        self.paintBoard.clear()
        self.progressBar.setProperty("value", 100)
        self.label.setText("Infer succeed")
        # 删除临时文件
        delete_file('./final.json')
        delete_file('./temp.png')

    def btn_func(self, btn):
        if btn == self.playbtn:
            self.label.setText("Play video")
            if self.progress_slider.value() == self.cap.get(7) - 1:
                self.slider_frame_num = 0
                self.progress_slider.setValue(self.slider_frame_num)
                self.time_label.setText('{}/{}'.format(self.slider_frame_num, self.cap.get(7)))
            self.timer_camera = QTimer()  # 定义定时器
            self.timer_camera.start(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)))
            self.slider_frame_num = self.progress_slider.value()
            self.timer_camera.timeout.connect(self.update_frame)

        elif btn == self.pushButton_2:
            self.label.setText("Stop video")
            self.slot_stop()

        elif btn == self.pushButton_4:
            self.label.setText("Choose video")
            self.select_video_path, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.mp4;;All Files(*)")
            Logging.debug("Select video file path:\t" + self.select_video_path)
            video_type_list = ["mp4", "MP4", "mov", "MOV", "avi", "AVI"]
            if self.select_video_path.split('/')[-1].split('.')[-1] in video_type_list:
                self.cap = cv2.VideoCapture(self.select_video_path)
                # 存所有frame
                self.save_temp_frame()
                Logging.debug("save temp frame done")
                self.progress_slider.setRange(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.slider_frame_num = 0
                self.update_frame()
            else:
                QMessageBox.information(self,
                                        "请选择正确的视频格式",
                                        "请选择正确的视频格式,目前默认支持mp4/MP4/mov/MOV/avi/AVI。",
                                        QMessageBox.Yes)

    def on_cbtn_eraser_clicked(self):
        self.label.setText("Eraser On")
        if self.cbtn_Eraser.isChecked():
            self.paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.paintBoard.EraserMode = False  # 退出橡皮擦模式

    def fill_color_list(self, combo_box):
        index_black = 0
        index = 0
        for color in self.colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            combo_box.addItem(QIcon(pix), None)
            combo_box.setIconSize(QSize(70, 20))
            combo_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        combo_box.setCurrentIndex(index_black)

    def on_pen_color_change(self):
        self.label.setText("Change pen color")
        color_index = self.comboBox_penColor.currentIndex()
        color_str = self.colorList[color_index]

        self.paintBoard.change_pen_color(color_str)

    # 拖拽进度条
    def update_video_position_func(self):
        self.label.setText("Change slider position")
        self.slider_frame_num = self.progress_slider.value()
        self.slot_stop()
        self.update_frame()
        self.progress_slider.setValue(self.slider_frame_num)
        self.time_label.setText('{}/{}'.format(self.slider_frame_num, self.cap.get(7)))

    def save_temp_frame(self):
        _, self.all_frames = load_video(self.select_video_path, 480)

    def slot_stop(self):
        if self.cap != []:
            self.timer_camera.stop()  # 停止计时器
        else:
            QMessageBox.warning(self, "Warming", "Push the left upper corner button to Quit.",
                                QMessageBox.Yes)

    def update_frame(self):
        self.progress_slider.setValue(self.slider_frame_num)
        self.slider_frame_num = self.progress_slider.value()
        self.frame = self.all_frames[self.slider_frame_num]
        frame = self.frame
        height, width, bytes_per_component = frame.shape
        bytes_per_line = bytes_per_component * width
        q_image = QImage(frame.data, width, height, bytes_per_line,
                         QImage.Format_RGB888).scaled(self.picturelabel.width(), self.picturelabel.height())
        self.picturelabel.setPixmap(QPixmap.fromImage(q_image))
        self.slider_frame_num = self.slider_frame_num + 1
        self.time_label.setText('{}/{}'.format(self.slider_frame_num, self.cap.get(7)))
        if self.progress_slider.value() == self.cap.get(7) - 1:
            self.slot_stop()
