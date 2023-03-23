#import Full_Path as FP
from sys import argv, exit
import threading
from PyQt5 import uic ,QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton , QLabel , QFileDialog, QCheckBox
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt
import cv2, imutils
import time
import numpy as np
import pyshine as ps
from ultralytics import YOLO
import Full_Path
import time
import skvideo.io

root = ""
'''
TO-DO : for @gemy 3shan y3rf m7tagen n3mel eh xd
        - Display Anomaly Score on Scree
        - Fix saving video with Yolo prediction
        - Seperate between anomaly prediction thread and displaying video thread
'''

# Create a subclass of the QWidget class to represent the main window
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)
        self.setWindowIcon(QtGui.QIcon('logo.ico'))
        self.setWindowTitle("Surveillance System")
        self.initialize_button()
        self.started = False
        self.score=0
        self.checkboxes = {
            "camera0": self.findChild(QCheckBox, "checkBox_0"),
            "camera1": self.findChild(QCheckBox, "checkBox_1"),
            "camera2": self.findChild(QCheckBox, "checkBox_2"),
            "camera3": self.findChild(QCheckBox, "checkBox_3"),
            "camera4": self.findChild(QCheckBox, "checkBox_4"),
            "camera5": self.findChild(QCheckBox, "checkBox_5"),
        }
        #self.anomaly_score = Full_Path.anomaly()

    def initialize_button(self):
        self.pushButton = self.findChild(QPushButton, "pushButton")
        self.pushButton.clicked.connect(self.pushButton_clicked)
        
    def pushButton_clicked(self):
        if self.started:
            self.started=False
            self.pushButton.setText('Start')
        else:
            self.started=True
            self.pushButton.setText('Stop')
    
        self.liveViewCamera0 = threading.Thread(target=self.loadVideo,  args=('liveView_0',f"testVids//anomaly1.mp4", self.checkboxes["camera0"].isChecked()))
        self.liveViewCamera0.start()
        self.liveViewCamera1 = threading.Thread(target=self.loadVideo,  args=('liveView_1',f"testVids//normal1.avi", self.checkboxes["camera1"].isChecked()))
        self.liveViewCamera1.start()
        self.liveViewCamera2 = threading.Thread(target=self.loadVideo,  args=('liveView_2',f"testVids//normal2.mp4", self.checkboxes["camera2"].isChecked()))
        self.liveViewCamera2.start()      
        self.liveViewCamera3 = threading.Thread(target=self.loadVideo,  args=('liveView_3',f"testVids//normal3.mp4", self.checkboxes["camera3"].isChecked()))
        self.liveViewCamera3.start()   
        self.liveViewCamera4 = threading.Thread(target=self.loadVideo,  args=('liveView_4',f"testVids//normal4.avi", self.checkboxes["camera4"].isChecked()))
        self.liveViewCamera4.start()
        self.liveViewCamera5 = threading.Thread(target=self.loadVideo,  args=('liveView_5',f"testVids//normal5.avi", self.checkboxes["camera5"].isChecked()))
        self.liveViewCamera5.start()
        
    def loadVideo(self, video_label, video_source, enabled):
        if not enabled:
            return

        vid = cv2.VideoCapture(video_source)
        count = 0
        model = YOLO("yolov8n.pt")
        num_objects = 0
        frame_buffer = []
        while vid.isOpened() and self.started:
            QtWidgets.QApplication.processEvents()
            ret, image = vid.read()
            if not ret:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            image = imutils.resize(image, height=480)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_buffer.append(image)
            if len(frame_buffer) == 24:
                video_frames = np.stack(frame_buffer, axis=0)
                #num_objects = len(model.predict(image, classes=0,verbose = False)[0])
                skvideo.io.vwrite(root + "Videos/sample.avi", frame_buffer)
                self.score = Full_Path.anomaly(frame_buffer)
                frame_buffer = []
                print(self.score)
            text = "Number of people : " + str(num_objects)
            #the anomaly model thinks ppl count is anomaly
            #image = ps.putBText(image,text,text_offset_x=30,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0,background_RGB=(228,20,222),text_RGB=(255,255,255))
            frame = image

            qimage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            label = self.findChild(QLabel, video_label)
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            count += 1

        vid.release()


# Create a QApplication instance and show the window
if __name__ == '__main__':
    app = QApplication(argv)
    window = MyWindow()
    window.show()
    exit(app.exec_())
