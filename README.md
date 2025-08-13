# Face Mask Detection - End-to-End Project

This project detects whether a person is wearing a **face mask** or **not** in real-time using **Computer Vision** and **Deep Learning**.  
It uses a custom dataset with two classes:  
- `with_mask`  
- `without_mask`  

The model is trained using **TensorFlow/Keras** and deployed with **OpenCV** for live video detection.

---

## Project Structure


## 🚀 Features
- **Custom dataset support**.
- **Real-time face mask detection** using a webcam.
- **Train/test split automation**.
- **CNN model with high accuracy**.
- **Easy deployment in Colab or VS Code**.

---

## 🛠 Tech Stack
- **Python 3.8+**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy, Matplotlib**
- **IPython** (for VS Code / Colab interactive mode)

---

## 📦 Installation

### 1. Clone the Repository

git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection

### 2️. Install Dependencies

pip install -r requirements.txt

### 3. Dataset Preparation
Place your raw dataset in my_data/mask and my_data/no_mask.
Run the script to split into train and test:

python split_data.py

### 4. Model Training
Train the CNN model:

python train_model.py

### 5. Real-Time Detection
Run live detection with webcam:

python main.py

## Future Improvements
- Improve accuracy with MobileNetV2 or EfficientNet.
- Add multi-person detection in a single frame.
- Deploy as a web app using Streamlit or Flask.

## License
This project is open-source under the MIT License.

## Author
- Name: Nikhil Mahesh
- E-mail: nikhilmahesh89@gmail.com
