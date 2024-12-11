import sys
import cv2
import pandas as pd
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QScrollArea, QSizePolicy, QFrame, QPushButton, QStatusBar)
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt6.QtCore import QTimer, Qt, QSize

class StyledLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLabel {
                color: #ffffff;
                padding: 8px;
                border-radius: 4px;
                background-color: #2c2c2c;
            }
        """)

class ProductCard(QFrame):
    def __init__(self, product_info, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border-radius: 8px;
                border: 1px solid #404040;
                margin: 5px;
                padding: 15px;
            }
            QFrame:hover {
                border: 1px solid #505050;
                background-color: #363636;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create info grid
        info_layout = QVBoxLayout()
        
        # Helper function to add info rows
        def add_info_row(label, value, highlight=False):
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_widget.setLayout(row_layout)
            
            label_widget = QLabel(label)
            label_widget.setStyleSheet("color: #888888; font-weight: bold;")
            label_widget.setFixedWidth(120)
            
            value_widget = QLabel(str(value))
            if highlight:
                value_widget.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
            else:
                value_widget.setStyleSheet("color: #ffffff;")
            
            row_layout.addWidget(label_widget)
            row_layout.addWidget(value_widget)
            row_layout.addStretch()
            
            info_layout.addWidget(row_widget)
        
        # Add horizontal line
        def add_separator():
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setStyleSheet("background-color: #404040;")
            info_layout.addWidget(line)
        
        # Brand Name as title
        title = QLabel(product_info['Band Name'])
        title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; padding-bottom: 5px;")
        layout.addWidget(title)
        
        add_separator()
        
        # Add all product details (excluding Class)
        add_info_row("Brand Details:", product_info.get('Brand Details', 'N/A'))
        add_info_row("MRP:", f"₹{product_info['MRP']:.2f}", highlight=True)
        add_info_row("Expiry Date:", product_info.get('Expiry Date', 'N/A'))
        add_info_row("Mfg. Date:", product_info.get('Manufacture Date', 'N/A'))
        
        layout.addLayout(info_layout)

class ProductDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Product Detection System")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #1a237e;
                color: #666666;
            }
            QStatusBar {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QStatusBar::item {
                border: none;
            }
        """)

        # Load model and data
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp10/weights/best.pt')
        self.product_data = pd.read_csv('updated_product_data.csv')

        # Set up camera
        self.video_url = 'http://<ur ip web cam address>/video'
        self.cap = cv2.VideoCapture(self.video_url)

        # Create main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Left panel for camera
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Camera title
        camera_title = StyledLabel("Live Detection Feed")
        camera_title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(camera_title)

        # Camera frame with border
        camera_frame = QFrame()
        camera_frame.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border-radius: 10px;
                border: 1px solid #404040;
                padding: 10px;
            }
        """)
        camera_layout = QVBoxLayout()
        camera_frame.setLayout(camera_layout)
        
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(720, 540)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 1px solid #404040;")
        camera_layout.addWidget(self.camera_label)
        
        left_layout.addWidget(camera_frame)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("▶ Start Detection")
        self.stop_button = QPushButton("⏹ Stop Detection")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #b71c1c;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)

        # Right panel for product details
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Details title
        details_title = StyledLabel("Product Details")
        details_title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        details_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(details_title)

        # Scroll area for product details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2c2c2c;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #404040;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #505050;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.product_details_widget = QWidget()
        self.product_details_layout = QVBoxLayout()
        self.product_details_layout.setSpacing(10)
        self.product_details_layout.setContentsMargins(10, 10, 10, 10)
        self.product_details_widget.setLayout(self.product_details_layout)
        scroll_area.setWidget(self.product_details_widget)
        right_layout.addWidget(scroll_area)

        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=60)
        main_layout.addWidget(right_panel, stretch=40)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("System Ready")

        # Set up timer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Connect buttons
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

    def update_status(self, message):
        self.status_bar.showMessage(f"Status: {message}")

    def start_detection(self):
        self.timer.start()
        self.update_status("Detection Active")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_detection(self):
        self.timer.stop()
        self.update_status("Detection Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Perform detection
            results = self.model(frame)

            # Clear previous product details
            for i in reversed(range(self.product_details_layout.count())): 
                self.product_details_layout.itemAt(i).widget().setParent(None)

            # Process detections
            detected_classes = []
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, class_id = detection[:6]
                class_id = int(class_id)
                class_name = self.model.names[class_id]
                
                if class_name not in detected_classes:
                    detected_classes.append(class_name)

                # Draw bounding box with confidence
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label with dark background for better visibility
                band_name = self.product_data[self.product_data['Class'] == class_name].iloc[0]['Band Name']
                label = f"{band_name} ({conf:.2f})"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1)-25), (int(x1)+label_w+10, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1)+5, int(y1)-7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Add product details
            for class_name in detected_classes:
                self.add_product_details(class_name)

            # Convert frame to QImage and display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def add_product_details(self, class_name):
        matched_row = self.product_data[self.product_data['Class'] == class_name]
        if not matched_row.empty:
            product_card = ProductCard(matched_row.iloc[0])
            self.product_details_layout.addWidget(product_card)

    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ProductDetectionGUI()
    window.show()
    sys.exit(app.exec())
