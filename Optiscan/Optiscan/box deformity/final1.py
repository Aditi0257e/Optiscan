from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QFrame, QPushButton, QMessageBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
import sys
import cv2
import torch
import pandas as pd
from pyzbar import pyzbar
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# ... (keep the parse_code128 and get_product_info_from_csv functions as they are) ...
def parse_code128(barcode):
    barcode_len = len(barcode)
    if barcode_len >= 9:
        identifier = barcode[:4]
        serial_number = barcode[4:9]
        return {
            "Identifier": identifier,
            "Serial Number": serial_number,
        }
    else:
        print(f"[INFO] Code 128 barcode does not match expected format: {barcode}")
        return None

def get_product_info_from_csv(barcode, barcodeType, csv_data):
    if barcodeType == "CODE128":
        code128_details = parse_code128(barcode)
        if code128_details:
            try:
                identifier = int(code128_details['Identifier'])
                serial_number = int(code128_details['Serial Number'])
                matching_row = csv_data[
                    (csv_data['identifier'] == identifier) & 
                    (csv_data['serial_number'] == serial_number)
                ]
                if not matching_row.empty:
                    product_info = matching_row.iloc[0]
                    product_name = f"Product {product_info['identifier']}"
                    brand_name = product_info['brand_name']
                    mfg_date = product_info['mfg_date']
                    expiry_date = product_info['expiry_date']
                    MRP = product_info['MRP']
                    return product_name, brand_name, mfg_date, expiry_date, MRP
                else:
                    print(f"[INFO] No matching product found in CSV for barcode: {barcode}")
                    return None, None, None, None, None
            except ValueError:
                print("[ERROR] Failed to parse barcode to integer")
                return None, None, None, None, None
        else:
            return None, None, None, None, None
    else:
        return None, None, None, None, None

class CameraThread(QThread):
    update_frame = pyqtSignal(QImage, str)

    def __init__(self, camera_url, is_barcode_camera=False):
        super().__init__()
        self.camera_url = camera_url
        self.is_barcode_camera = is_barcode_camera
        self.cap = cv2.VideoCapture(self.camera_url)
        
        if self.is_barcode_camera:
            self.csv_data = pd.read_csv('updated_sample_barcode_data_with_mrp.csv')
            self.csv_data['identifier'] = self.csv_data['identifier'].astype(int)
            self.csv_data['serial_number'] = self.csv_data['serial_number'].astype(int)
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='demo_cardboard/yolov5/runs/train/exp8/weights/best.pt', device='cpu')
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='demo_cardboard/yolov5/runs/train/exp8/weights/best.pt', device='cpu')
        
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.model.classes = None
        self.model.max_det = 20

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                if self.is_barcode_camera:
                    frame, info = self.process_barcode_frame(frame)
                else:
                    frame, info = self.process_deformity_frame(frame)
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_image = qt_image.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.update_frame.emit(scaled_image, info)

    def process_barcode_frame(self, frame):
    # Perform barcode detection
        results = self.model(frame)  # YOLOv5 deformity detection model
        results.render()  # Render detections on the frame
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        barcodes = pyzbar.decode(gray)
    
        info = ""
    
    # Check for barcode detections and process them
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
        
        # Retrieve product information from CSV
            product_name, brand, mfg_date, expiry_date, mrp = get_product_info_from_csv(barcodeData, barcodeType, self.csv_data)
            text = f"{barcodeData}\n ({barcodeType})"
            if product_name and brand:
                text += f"\nProduct: {product_name}\nBrand: {brand}\nMFG Date: {mfg_date}\nEXP Date: {expiry_date}\nMRP: {mrp}"
        
        # Perform deformity detection using YOLOv5 on the frame
            deformity_detected = False
            for *box, conf, cls in results.xyxy[0]:
                label = f"{results.names[int(cls)]}: {conf:.2f}"
                if "deformity" in results.names[int(cls)].lower():
                    deformity_detected = True
                    break
        
        # Determine if the box is "Good" or "Bad" based on deformity detection
            if deformity_detected:
                text += "\nBox Condition: Bad Box"
            else:
                text += "\nBox Condition: Good Box"
        
            cv2.putText(frame, barcodeData, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            info += f"{text}\n\n"
        
        # Print the complete information in the console as well
            print(f"Barcode Detected: {barcodeData}")
            print(f"Type: {barcodeType}")
            if product_name and brand:
                print(f"Product: {product_name}\n")
                print(f"Brand: {brand}\n")
                print(f"Manufacturing Date: {mfg_date}\n")
                print(f"Expiry Date: {expiry_date}\n")
                print(f"MRP: {mrp}")
            print(f"Box Condition: {'Bad Box' if deformity_detected else 'Good Box'}\n")
            print("-" * 40)
    
        return frame, info


    def process_deformity_frame(self, frame):
        # ... (keep the existing process_deformity_frame method) ...
        results = self.model(frame)
        info = "Detected objects:\n"
        for *box, conf, cls in results.xyxy[0]:
            label = f"{results.names[int(cls)]}: {conf:.2f}"
            info += f"{label}\n"
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, info

class CameraWidget(QFrame):
    def __init__(self, camera_url, is_barcode_camera=False):
        super().__init__()
        self.is_barcode_camera = is_barcode_camera
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setMidLineWidth(1)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Create image label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create info display frame with different styles based on camera type
        info_frame = QFrame()
        if self.is_barcode_camera:
            # Barcode camera style (frame 1)
            info_frame.setStyleSheet("""
                QFrame {
                    background-color: rgba(40, 40, 40, 200);
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
        else:
            # Deformity detection camera style (frame 2)
            info_frame.setStyleSheet("""
                QFrame {
                    background-color: rgba(30, 34, 39, 230);
                    border-radius: 15px;
                    margin: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
            """)
        
        info_layout = QVBoxLayout(info_frame)
        
        # Create header for frame 2
        if not self.is_barcode_camera:
            header_label = QLabel("Deformity Detection Analysis")
            header_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    background-color: rgba(61, 90, 128, 0.3);
                    border-radius: 5px;
                }
            """)
            header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_layout.addWidget(header_label)
        
        # Create and style info label
        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info_label.setFont(QFont("Arial", 11))
        self.info_label.setWordWrap(True)
        
        if self.is_barcode_camera:
            # Barcode camera info style
            self.info_label.setStyleSheet("""
                QLabel {
                    color: white;
                    padding: 15px;
                    line-height: 1.5;
                }
            """)
        else:
            # Deformity detection info style
            self.info_label.setStyleSheet("""
                QLabel {
                    color: white;
                    padding: 20px;
                    line-height: 1.8;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
        
        # Add confidence meter for frame 2
        if not self.is_barcode_camera:
            self.confidence_meter = QProgressBar()
            self.confidence_meter.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #2C3E50;
                    border-radius: 5px;
                    text-align: center;
                    height: 25px;
                    background-color: rgba(44, 62, 80, 0.7);
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3498DB, stop:1 #2ECC71);
                    border-radius: 3px;
                }
            """)
            self.confidence_meter.setTextVisible(True)
            info_layout.addWidget(self.confidence_meter)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Create and style buttons
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.discard_button = QPushButton("Discard")
        self.discard_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        # Connect button signals
        self.discard_button.clicked.connect(self.show_discard_popup)
        
        # Add buttons to layout
        buttons_layout.addWidget(self.next_button)
        buttons_layout.addWidget(self.discard_button)
        buttons_layout.setSpacing(20)
        buttons_layout.setContentsMargins(10, 0, 10, 10)
        
        # Add widgets to layouts
        info_layout.addWidget(self.info_label)
        info_layout.addLayout(buttons_layout)
        
        layout.addWidget(self.image_label)
        layout.addWidget(info_frame)
        self.setLayout(layout)
        
        # Start camera thread
        self.thread = CameraThread(camera_url, is_barcode_camera)
        self.thread.update_frame.connect(self.update_frame)
        self.thread.start()

    def update_frame(self, image, info):
        self.image_label.setPixmap(QPixmap.fromImage(image))
        
        if self.is_barcode_camera:
            formatted_info = self.format_barcode_info(info)
        else:
            formatted_info = self.format_deformity_info(info)
            
        self.info_label.setText(formatted_info)

    def format_deformity_info(self, info):
        if not info:
            return ""
            
        # Extract confidence values from detections
        confidence_values = []
        formatted_lines = ['<div style="background-color: rgba(52, 73, 94, 0.5); padding: 10px; border-radius: 5px;">']
        
        for line in info.split('\n'):
            if line.startswith("Detected objects:"):
                formatted_lines.append('<span style="color: #3498DB; font-size: 14px; font-weight: bold;">Detection Results:</span><br>')
            elif ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    obj_name = parts[0].strip()
                    confidence = float(parts[1].strip())
                    confidence_values.append(confidence)
                    
                    # Color code based on confidence
                    if confidence > 0.8:
                        color = "#2ECC71"  # Green for high confidence
                    elif confidence > 0.5:
                        color = "#F1C40F"  # Yellow for medium confidence
                    else:
                        color = "#E74C3C"  # Red for low confidence
                    
                    formatted_lines.append(
                        f'<div style="margin: 5px 0; padding: 8px; background-color: rgba(255, 255, 255, 0.1); '
                        f'border-radius: 4px;">'
                        f'<span style="color: {color};">■</span> '
                        f'<span style="color: #ECF0F1; font-weight: bold;">{obj_name}</span>: '
                        f'<span style="color: {color}; float: right;">{confidence:.2%}</span>'
                        f'</div>'
                    )

        formatted_lines.append('</div>')
        
        # Update confidence meter if values exist
        if confidence_values and hasattr(self, 'confidence_meter'):
            avg_confidence = sum(confidence_values) / len(confidence_values)
            self.confidence_meter.setValue(int(avg_confidence * 100))
            self.confidence_meter.setFormat(f"Average Confidence: {avg_confidence:.1%}")
        
        return "".join(formatted_lines)

    def format_barcode_info(self, info):
        # Existing barcode formatting logic remains the same
        if not info:
            return ""
            
        lines = info.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith("Box Condition:"):
                condition = line.split(":")[1].strip()
                color = "#4CAF50" if "Good" in condition else "#f44336"
                formatted_lines.append(f'<span style="color: {color}; font-weight: bold;">{line}</span>')
            elif ":" in line:
                label, value = line.split(":", 1)
                formatted_lines.append(f'<b>{label}:</b>{value}')
            else:
                formatted_lines.append(line)
                
        return "<br>".join(formatted_lines)

    def show_discard_popup(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Discarded Successfully")
        msg.setWindowTitle("Success")
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #333333;
            }
            QMessageBox QLabel {
                color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-size: 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        msg.exec()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual-Camera Detection System")
        self.setGeometry(100, 100, 1366, 768)  # Adjusted for a typical laptop screen

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create two camera widgets
        self.camera1 = CameraWidget("http://<1st camera>/video", is_barcode_camera=True)
        self.camera2 = CameraWidget("http://<2nd camera>/video")

        main_layout.addWidget(self.camera1)
        main_layout.addWidget(self.camera2)

        # Set a dark theme
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(dark_palette)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
