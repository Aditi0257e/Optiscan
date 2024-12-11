The Box Deformity Detection Model is based on the YOLOv5 architecture and identifies tampered or deformed packaging by analyzing the shape, size, and condition of the box.

## Steps to Run:
1. Install YOLOv5:
  Follow the instructions to clone and set up YOLOv5:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
pip install pyzar
```
2. Dataset
   - The dataset used for this project is too large to host on GitHub. You can download it or create ur own dataset
   - Gather all the images you want to use for training. Ensure they are in JPEG (.jpg) or PNG (.png) format.
   - Divide the dataset into training, validation, and optionally test sets. Typically, you can use an 80/20 or 70/20/10 split.
   ```bash
     /dataset
        /images
          /train
          /val
        /labels
          /train
          /val
   ```
   - Label the Images
      YOLOv5 requires labels in a specific format. You can use labeling tools like:
      - LabelImg
    - Each image must have a corresponding .txt file with the same name. Each line in the .txt file represents a bounding box and should follow this format:
      ```bash
        <class_id> <x_center> <y_center> <width> <height>
      ```
  3. Yaml file
     set up the yaml folder according to the custom_data.yaml
  4. train the model
     ```bash
         python train.py --img 640 --batch 16 --epochs 1 --data path/to/data.yaml --weights yolov5s.pt --cache
     ```
  6. After training, evaluate the model on the validation set:
     ```bash
         python val.py --data path/to/data.yaml --weights runs/train/exp/weights/best.pt --img 640
      ```
  5. run the model
     ```bash
         python detect.py --source path_to_images --weights path_to_weights --source http://<ip web cam address>/video
     ```
  6. run the integrated model
     ```bash
         //attach some barcode images to the box and run the final1.py
         python final1.py
     ```


       


     
