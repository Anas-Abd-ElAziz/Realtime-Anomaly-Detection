from sys import argv, exit
import threading
from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QCheckBox, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2, imutils
import time
import numpy as np
import pyshine as ps
from ultralytics import YOLO
from queue import Queue, Empty
import Full_Path as RTFM
from PIL import Image
import os
os.environ['QT_LOGGING_RULES'] = "qt.gui.icc=false"
root = ""

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)
        self.setWindowIcon(QtGui.QIcon('bus.jpg'))
        self.setWindowTitle("Surveillance System")
        self.initialize_button()
        self.started = False
        self.checkboxes = {
            "camera0": self.findChild(QCheckBox, "checkBox_0"),
            "camera1": self.findChild(QCheckBox, "checkBox_1"),
            "camera2": self.findChild(QCheckBox, "checkBox_2"),
            "camera3": self.findChild(QCheckBox, "checkBox_3"),
            "camera4": self.findChild(QCheckBox, "checkBox_4"),
            "camera5": self.findChild(QCheckBox, "checkBox_5"),
        }
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_model.predict(cv2.imread('logo.ico'), classes=0, verbose=False)


    def initialize_button(self):
        self.pushButton = self.findChild(QPushButton, "pushButton")
        self.pushButton.clicked.connect(self.pushButton_clicked)


    def pushButton_clicked(self):
        if self.started:
            self.started = False
            self.pushButton.setText('Start')
        else:
            self.started = True
            self.pushButton.setText('Stop')

        self.liveViewCamera0 = threading.Thread(target=self.loadVideo,  args=('liveView_0',f"testVids//normal1.avi", self.checkboxes["camera0"].isChecked(), self.yolo_model))
        self.liveViewCamera0.start()
        self.liveViewCamera1 = threading.Thread(target=self.loadVideo,  args=('liveView_1',f"testVids//anomaly1.mp4", self.checkboxes["camera1"].isChecked(), self.yolo_model))
        self.liveViewCamera1.start()
        self.liveViewCamera2 = threading.Thread(target=self.loadVideo,  args=('liveView_2',f"testVids//normal2.mp4", self.checkboxes["camera2"].isChecked(), self.yolo_model))
        self.liveViewCamera2.start()      
        self.liveViewCamera3 = threading.Thread(target=self.loadVideo,  args=('liveView_3',f"testVids//normal3.mp4", self.checkboxes["camera3"].isChecked(), self.yolo_model))
        self.liveViewCamera3.start()   
        self.liveViewCamera4 = threading.Thread(target=self.loadVideo,  args=('liveView_4',f"testVids//normal4.avi", self.checkboxes["camera4"].isChecked(), self.yolo_model))
        self.liveViewCamera4.start()
        self.liveViewCamera5 = threading.Thread(target=self.loadVideo,  args=('liveView_5',f"testVids//normal5.avi", self.checkboxes["camera5"].isChecked(), self.yolo_model))
        self.liveViewCamera5.start()

    def loadVideo(self, video_label, video_source, enabled, shared_yolo_model):
        if not enabled:
            return

        vid = cv2.VideoCapture(video_source)
        count = 0
        yolo_queue = Queue(maxsize=1)
        anomaly_queue = Queue(maxsize=1)

        def yolo_predict(image):
            nonlocal yolo_queue
            start_time = time.time()
            num_objects = len(shared_yolo_model.predict(image, classes=0, verbose=False)[0])
            end_time = time.time()
            print("Execution time:", end_time - start_time, "seconds")
            try:
                yolo_queue.put_nowait(num_objects)
            except Exception as e:
                pass
        
        def anomaly_predict(frame_buffer):
            nonlocal anomaly_queue
            anomaly_score = anomaly_score = np.mean(RTFM.anomaly(frame_buffer).tolist()[0])
            try:
                anomaly_queue.put_nowait(anomaly_score)
            except Exception as e:
                pass

        num_objects = 0
        anomaly_score = 0
        frame_buffer = []
        current_prediction=-1
        first_frame = True
        
        while vid.isOpened() and self.started:
            QtWidgets.QApplication.processEvents()
            ret, image = vid.read()
            if not ret:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frame_buffer.append(pil_img)
            image = imutils.resize(image, height=480)

            if len(frame_buffer) % 12 == 0:
                yolo_thread = threading.Thread(target=yolo_predict, args=(image,))
                yolo_thread.start()

            if len(frame_buffer) == 24:
                if current_prediction != anomaly_score:
                    anomaly_thread = threading.Thread(target=anomaly_predict, args=(frame_buffer,))
                    anomaly_thread.start()
                    current_prediction=anomaly_score
                frame_buffer = []

            try:
                num_objects = yolo_queue.get_nowait()
            except Empty:
                pass

            try:
                anomaly_score = anomaly_queue.get_nowait()
            except Empty:
                pass

            text = "Number of people : " + str(num_objects)
            image = ps.putBText(image, text, text_offset_x=30, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0, background_RGB=(228, 20, 222), text_RGB=(255, 255, 255))
            anomaly_state = 'Anomaly' if anomaly_score > 0.7 else 'Normal'
            text = "Anomaly State : " + str(anomaly_score)[:5] + " "+ anomaly_state
            image = ps.putBText(image, text, text_offset_x=30, text_offset_y=90, vspace=20, hspace=10, font_scale=1.0, background_RGB=(230, 230, 230), text_RGB=((10, 255, 10) if anomaly_state == 'Normal' else (255, 10, 10)))
            
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            label = self.findChild(QLabel, video_label)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

            time.sleep(0.024)  # add a delay of 42 milliseconds (1/24th of a second)
            
        vid.release()

# Create a QApplication instance and show the window
if __name__ == '__main__':
    app = QApplication(argv)
    window = Window()
    window.show()
    exit(app.exec_())
