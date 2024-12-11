import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QProgressBar, QFrame)
from PyQt6 import QtGui, QtCore
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, QRect

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

# ... [Keep the Model class as is] ...
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.alpha = 0.7
        self.base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
        
        self.base.fc = nn.Sequential()
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(128, 9)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1 = self.block2(x)
        y2 = self.block3(x)
        return y1, y2

class ColorfulProgressBar(QProgressBar):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        background_color = QColor(60, 60, 60)
        painter.setBrush(background_color)
        painter.drawRect(self.rect())

        # Progress
        progress = self.value() / (self.maximum() - self.minimum())
        progress_width = int(self.width() * progress)
        progress_rect = QRect(0, 0, progress_width, self.height())

        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(255, 0, 0))    # Red
        gradient.setColorAt(0.5, QColor(255, 255, 0))  # Yellow
        gradient.setColorAt(1, QColor(0, 255, 0))    # Green

        painter.setBrush(gradient)
        painter.drawRect(progress_rect)

        # Text
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.value()}%")

class FruitFreshnessGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fruit Freshness Detection")
        self.setFixedSize(800, 700)  # Increased height to accommodate rearranged elements
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model()
        self.model.load_state_dict(torch.load('fruit_fresh_model.pth', 
                                            map_location=self.device, 
                                            weights_only=True))
        self.model.eval()
        self.model = self.model.to(self.device)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)  # Add space between widgets

        # Create top panel for controls
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Connect/Disconnect button
        self.connect_button = QPushButton("Start Detection")
        self.connect_button.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_button)

        layout.addWidget(control_panel)

        # Create video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 10px;")
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Create result display
        self.result_label = QLabel("Fruit: -- | Freshness: --")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px 0;")
        layout.addWidget(self.result_label)

        # Add warning label for spoiled fruit
        self.warning_label = QLabel("")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF4136; margin: 10px 0;")
        layout.addWidget(self.warning_label)

        # Add colorful progress bar for freshness score
        self.freshness_bar = ColorfulProgressBar()
        self.freshness_bar.setRange(0, 100)
        self.freshness_bar.setTextVisible(False)
        self.freshness_bar.setFixedHeight(30)
        layout.addWidget(self.freshness_bar)

        # Initialize video capture variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Fruit class mapping
        self.fruit_classes = {
            0: "Apple", 1: "Banana", 2: "Cucumber", 3: "Okra", 
            4: "Orange", 5: "Potato", 6: "Tomato", 7: "Other"
        }

        # Add frame counter for processing
        self.frame_count = 0
        self.process_every_n_frames = 10

        # Set default camera URL
        self.camera_url = "http://192.168.0.248:8080/video"  # Replace with your default camera URL

    def toggle_connection(self):
        if self.timer.isActive():
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self):
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            self.result_label.setText("Error: Could not connect to camera")
            return

        self.connect_button.setText("Stop Detection")
        self.timer.start(30)

    def disconnect_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.connect_button.setText("Start Detection")
        self.video_label.clear()
        self.result_label.setText("Fruit: -- | Freshness: --")
        self.warning_label.setText("")
        self.freshness_bar.setValue(0)

    def preprocess_frame(self, frame):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])
        
        image = transform(frame)
        image = image.unsqueeze(0)
        return image

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            if self.frame_count % self.process_every_n_frames == 0:
                image = self.preprocess_frame(frame)
                image = image.to(self.device)
                
                with torch.no_grad():
                    fruit_pred, freshness_pred = self.model(image)
                    fruit_label = torch.argmax(fruit_pred, axis=1).item()
                    freshness_prob = F.softmax(freshness_pred, dim=1)
                    freshness_score = freshness_prob[0][1].item() * 100
                    freshness_label = torch.argmax(freshness_pred, axis=1).item()

                fruit_name = self.fruit_classes.get(fruit_label, "Unknown")
                freshness_status = "Fresh" if freshness_label == 1 else "Spoiled"
                self.result_label.setText(f"Fruit: {fruit_name} | Freshness: {freshness_status}")
                self.freshness_bar.setValue(int(freshness_score))

                if freshness_status == "Spoiled":
                    self.warning_label.setText(f"Warning: {fruit_name} may be spoiled!")
                else:
                    self.warning_label.setText("")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.disconnect_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FruitFreshnessGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()