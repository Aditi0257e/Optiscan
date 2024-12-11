# How to Run the Models
The Freshness Model is designed to assess the freshness of perishable products by analyzing their appearance. The model predicts the remaining shelf life based on visual cues.
## Steps to Run:
1. Install Dependencies:
  Install the necessary Python packages using the following command:
  ```bash
  pip install -r requirements.txt
  ```
  it will be nice if you install the dependencies in the virtual env 
``` bash
pip install virtualenv
python -m venv <your env name>
<your env name>/Scripts/activate
pip install -r requirements.txt
```
2. Dataset
   The dataset used for this project is too large to host on GitHub. You can download it from the following link:
    [Download Dataset from Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
   or from the hummingface

3. run the training model
   ```bash
   python freshness.py
   ```
4. run the model1.py
   before running the model1.py please make sure u provided the ip webcam ip address and correct path of your pretrained model
   ```bash
   python model1.py
   ```

   you can check ur camera of your phone whether it is linked perfectly or not by running the camera.py
5. use the gui
   ```bash
   python final3.py
   ```
 
   
