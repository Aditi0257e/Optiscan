# Required Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2  # For real-time camera feed
import numpy as np

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model architecture (make sure it matches your trained model)
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
            nn.Linear(128, 9)  # Change this to match the number of fruit classes
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # 2 classes: fresh/spoiled
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1 = self.block2(x)  # Fruit prediction
        y2 = self.block3(x)  # Freshness prediction
        return y1, y2

state_dict = torch.load('fruit_fresh_model.pth', map_location=device, weights_only=True)  # Use weights_only=True
model = Model()
model.load_state_dict(state_dict)
model.eval()  # Set model to evaluation mode
model = model.to(device)
# Preprocess Input Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Use correct mean and std values if different
    ])
    
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure it's 3-channel RGB
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Prediction Function
def predict_freshness(image_path):
    image = preprocess_image(image_path)
    image = image.to(device)
    
    with torch.no_grad():
        fruit_pred, freshness_pred = model(image)
        fruit_label = torch.argmax(fruit_pred, axis=1).item()  # Predicted fruit class
        freshness_label = torch.argmax(freshness_pred, axis=1).item()  # 0: spoiled, 1: fresh

    return fruit_label, freshness_label# Modify the preprocess_image function to accept a frame instead of a path
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert from OpenCV format to PIL
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Use correct mean and std
    ])
    
    image = transform(frame)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Modify the prediction function to accept a frame directly
def predict_freshness_from_frame(frame):
    image = preprocess_frame(frame)
    image = image.to(device)
    
    with torch.no_grad():
        fruit_pred, freshness_pred = model(image)
        fruit_label = torch.argmax(fruit_pred, axis=1).item()  # Predicted fruit class
        freshness_label = torch.argmax(freshness_pred, axis=1).item()  # 0: spoiled, 1: fresh

    return fruit_label, freshness_label


# Real-time Camera Capture Function
def capture_real_time_image():
    ip_webcam_url = 'http://<ur ip address>/video'  # Replace with your IP webcam URL
    cap = cv2.VideoCapture(ip_webcam_url)  # Open the IP webcam feed

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    frame_count = 0  # Counter to control frame processing rate
    process_every_n_frames = 5  # Process 1 out of every 5 frames
    
    while True:
        ret, frame = cap.read()
        if ret:
            # Process only every 5th frame
            if frame_count % process_every_n_frames == 0:
                # Run the prediction on the captured frame
                fruit_class, freshness_class = predict_freshness_from_frame(frame)
                freshness_status = 'Fresh' if freshness_class == 1 else 'Spoiled'
            
            # Display result on the frame (every frame)
            cv2.putText(frame, f"Freshness: {freshness_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Real-time Freshness Prediction', frame)

        # Increment the frame count
        frame_count += 1
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the real-time capture and prediction
capture_real_time_image()
