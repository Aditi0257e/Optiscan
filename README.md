# **Optiscan**
This project aims to revolutionize the quality inspection process for India's largest ecommerce company by leveraging camera vision technology. The system is designed to assess the shipment quality and quantity efficiently using image processing, machine learning, and sensor integration.
---
## **Project Overview**
The system utilizes advanced Smart Vision Technology to automate quality control processes. By analyzing images of products, it ensures that the products meet predefined standards for quality, quantity, and packaging integrity. This is achieved through a combination of camera vision, OCR, and machine learning models that detect defects, identify labels, and verify product freshness.


https://github.com/user-attachments/assets/85204c5c-db9d-46f1-aec2-7ac7e21db849


### Key Features:
1. **Image Acquisition:**
  - High-resolution cameras and optimized lighting setups to capture clear images of products on conveyor belts or in inventory.
2. **Image Preprocessing:**
  - Filters and normalization techniques to enhance the quality of images for further processing.
3. **Feature Extraction:**
  - Detect and recognize product details using OCR to extract text such as labels, descriptions, and expiration dates.
  - Geometric analysis for defect detection and color/texture analysis for freshness prediction.
4. **Classification and Decision-Making:**
  - Machine learning models like CNNs and SVMs to classify products and detect quality issues.
  - Deep learning techniques to continuously improve accuracy based on large datasets.
5. **Output and Feedback:**
  - Real-time product identification and feedback to operators for defective products.
  - Integration with inventory systems for automatic updates.
6. **Integration:**
  - Automated sorting using robotic arms or conveyors based on quality assessments.
  - Data logging and feedback loops for continuous improvement.

### **Use Cases:**
- ***OCR for Labels and Expiration Dates:*** Automatically extract details from packaging labels and verify expiration dates for product quality assurance.
- ***Fresh Produce Inspection***: Analyze the freshness of fruits, vegetables, and other perishable items by detecting visual cues.
- ***Packaging and Item Count Verification***: Ensure packaging integrity and count the number of products accurately.

### **Challenges**
- ***Lighting and Environmental Factors:*** Variations in lighting conditions may impact the accuracy of image recognition.
- ***Product Variability:*** Diverse sizes, shapes, and colors of products add complexity to the inspection process.

### **Applications**
This system can be applied to the ecommerce industry for:
- **Product Identification:** Identifying items and verifying quantities.
- **Packaging Inspection:** Ensuring packaging correctness and detecting tampering.
- **Quality Control for Fresh Produce:** Inspecting fruits and vegetables for spoilage or defects.
- **Inventory Management:** Automated tracking of stock in bins or on shelves.
---
### Current Status
The system is composed of three distinct models:
- Freshness Model
- OCR and Barcode Scanner
- Box Deformity Detection Model
These models are not yet integrated and must be run independently.
instructions for running each model separately are given in each folder.
You are requested to go through the folders and read the readme files to understand it better !!!

### Future Work
- **Model Integration:** We are working on integrating the three models to run simultaneously and generate a unified output.
- **Real-Time Feedback:** The integrated system will eventually provide real-time quality assessments and feedback for automated systems.
--

# Contributing
Feel free to contribute to any part of the project. You can work on integration or model improvements. Please follow the code guidelines and write tests for any new functionality.

