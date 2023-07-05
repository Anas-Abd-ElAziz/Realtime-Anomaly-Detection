from sys import argv, exit
import threading
from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QCheckBox, QWidget, QSlider, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2, imutils
import time
import numpy as np
import pyshine as ps
from ultralytics import YOLO
from queue import Queue, Empty
import Full_Path as RTFM
from PIL import Image, ImageGrab
import os
from playsound import playsound
import pygame
import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ['QT_LOGGING_RULES'] = "qt.gui.icc=false"


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)
        self.setWindowIcon(QtGui.QIcon('assets//logo.ico'))
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
            "notification": self.findChild(QCheckBox, "checkBox_12"),
            "screenshot": self.findChild(QCheckBox, "checkBox_13"),
            "alert": self.findChild(QCheckBox, "checkBox_14"),
            "bboxes": self.findChild(QCheckBox, "checkBox_15"),
        }
        self.anomalySlider = self.findChild(QSlider, "anomalySlider")
        self.anomalySliderValue = self.findChild(QLabel, "anomalySliderValue")
        self.anomalySlider.valueChanged.connect(self.update_anomaly_slider_value)
        self.anomalySensitivity = 0.7

        self.yoloComboBox = self.findChild(QComboBox, "yoloComboBox")
        self.yoloComboBox.addItems(['Better performance', 'Better accuracy'])

    def update_anomaly_slider_value(self, value):
        self.anomalySensitivity = float(value)/100
        self.anomalySliderValue.setText(str(self.anomalySensitivity))


    def closeEvent(self, event):
        self.started = False
        event.accept()

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

        self.liveViewCamera0 = threading.Thread(target=self.loadVideo,  args=('liveView_0',f"testVids//normal1.avi", self.checkboxes["camera0"].isChecked()))
        self.liveViewCamera0.start()
        self.liveViewCamera1 = threading.Thread(target=self.loadVideo,  args=('liveView_1',f"testVids//anomaly1.mp4", self.checkboxes["camera1"].isChecked()))
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
        yolo_queue = Queue(maxsize=1)
        anomaly_queue = Queue(maxsize=1)

        yolo_model = YOLO("yolo//yolov8n.pt") if self.yoloComboBox.currentText() == 'Better performance' else YOLO("yolo//yolov8s.pt")
        

        def yolo_predict(image):
            nonlocal yolo_queue
            try:
                bbox = yolo_model.predict(image, classes=0, verbose=False)[0]
            except Exception as e:
                pass

            try:
                yolo_queue.put_nowait(bbox)
            except Exception as e:
                pass
        
        def anomaly_predict(frame_buffer):
            nonlocal anomaly_queue
            anomaly_score = anomaly_score = np.mean(RTFM.anomaly(frame_buffer).tolist()[0])
            try:
                anomaly_queue.put_nowait(anomaly_score)
            except Exception as e:
                pass
        
        def play_sound_alert():
            pygame.mixer.init()
            pygame.mixer.music.load("assets//alert.mp3")
            pygame.mixer.music.play()
            pygame.time.wait(3000)
            pygame.mixer.music.stop()
            pygame.mixer.quit()

        def send_email_alert():
            try:
                email_from = "surveillance.system.ai@gmail.com"
                email_to = "gemy212121@gmail.com"
                subject = "Anomaly Alert!"
                body = "An anomaly has been detected."
                
                msg = MIMEMultipart()
                msg['From'] = email_from
                msg['To'] = email_to
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(email_from, "nwgevudksgixckdn")
                server.send_message(msg)
                server.quit()
                
                print("Email alert sent!")
            except Exception as e:
                print("Error sending email alert:", str(e))

        num_objects = 0
        bbox = None
        anomaly_score = 0
        frame_buffer = []
        current_prediction=-1
        
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

            if self.checkboxes["bboxes"].isChecked():
                bbox = yolo_model.predict(image, classes=0, verbose=False)[0]
                num_objects = len(bbox) if bbox != None else 0
                if bbox != None:
                    for result in bbox:
                        boxes = result.boxes
                        for box in boxes:
                            x, y, w, h = box.xywh[0]
                            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif len(frame_buffer) % 12 == 0 and not self.checkboxes["bboxes"].isChecked():
                yolo_thread = threading.Thread(target=yolo_predict, args=(image,))
                yolo_thread.start()
                
            if len(frame_buffer) == 24:
                if current_prediction != anomaly_score:
                    anomaly_thread = threading.Thread(target=anomaly_predict, args=(frame_buffer,))
                    anomaly_thread.start()
                    current_prediction=anomaly_score
                frame_buffer = []

            try:
                bbox = yolo_queue.get_nowait()
                num_objects = len(bbox) if bbox != None else 0
            except Empty:
                pass

            try:
                anomaly_score = anomaly_queue.get_nowait()
            except Empty:
                pass
          
            text = "Number of people : " + str(num_objects)
            image = ps.putBText(image, text, text_offset_x=30, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0, background_RGB=(228, 20, 222), text_RGB=(255, 255, 255))
            anomaly_state = 'Anomaly' if anomaly_score > self.anomalySensitivity else 'Normal'
            text = "Anomaly State : " + str(anomaly_score)[:5] + " "+ anomaly_state
            image = ps.putBText(image, text, text_offset_x=30, text_offset_y=90, vspace=20, hspace=10, font_scale=1.0, background_RGB=(230, 230, 230), text_RGB=((10, 255, 10) if anomaly_state == 'Normal' else (255, 10, 10)))            
            label = self.centralwidget.findChild(QLabel, video_label)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage) 
            pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

            if anomaly_score > self.anomalySensitivity:
                if self.checkboxes["alert"].isChecked():
                    self.checkboxes["alert"].setChecked(False)
                    alert_thread = threading.Thread(target=play_sound_alert)
                    alert_thread.start()
                if self.checkboxes["screenshot"].isChecked():
                    self.checkboxes["screenshot"].setChecked(False)
                    screenshot = ImageGrab.grab()
                    screenshot.save(f'screenshots/screenshot_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png')
                if self.checkboxes["notification"].isChecked():
                    self.checkboxes["notification"].setChecked(False)
                    send_email_thread = threading.Thread(target=send_email_alert)
                    send_email_thread.start()

            time.sleep(0.024)

        vid.release()

# Create a QApplication instance and show the window
if __name__ == '__main__':
    app = QApplication(argv)
    window = Window()
    window.show()
    exit(app.exec_())
