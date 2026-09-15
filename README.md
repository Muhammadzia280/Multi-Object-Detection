# 🚀 Multi-Object Detection System using YOLOv8

A real-time **Multi-Object Detection System** built using **YOLOv8, Python, OpenCV, and Streamlit**.

This project detects and classifies multiple objects from images, videos, and real-time webcam input through an interactive web interface.

## 🎯 Detected Classes

| ID | Class |
|---:|---|
| 0 | Person |
| 1 | Car |
| 2 | Bike |
| 3 | Bus |
| 4 | Dog |
| 5 | Cat |
| 6 | Mask |
| 7 | No_Mask |

## ✨ Features

- 🤖 YOLOv8-based object detection
- 🎯 8-class object detection
- 🖼️ Image detection
- 🎥 Video detection
- 📷 Real-time webcam detection
- 🌐 Streamlit web interface
- 📥 Downloadable detection videos
- 🎨 Custom detection colors
- ⚡ GPU-supported training

## 🧠 Technologies

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- PyTorch

## 📂 Project Structure

```text
Multi_Object_Detection/
│
├── app.py
├── camera.py
├── image_detection.py
├── video_detection.py
├── best.pt
├── requirements.txt
├── Untitled25.ipynb
└── README.md
```

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Precision | 77.8% |
| Recall | 70.4% |
| mAP@50 | 76.7% |
| mAP@50-95 | 54.6% |

## 🏋️ Training Configuration

```text
Model: YOLOv8n
Epochs: 50
Image Size: 640
Batch Size: 16
Classes: 8
```

## 🖥️ Detection Modes

### 🖼️ Image Detection
Upload an image and detect supported objects with bounding boxes and class labels.

### 🎥 Video Detection
Upload a video and generate a processed video containing object detections.

### 📷 Real-Time Webcam
Use a webcam for real-time multi-object detection.

## 🎨 Detection Colors

- 🔵 Person, Car, Bike, Bus, Dog, Cat
- 🟢 Mask
- 🔴 No_Mask

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Muhammadzia280/Multi-Object-Detection.git
```

Go to the project directory:

```bash
cd Multi-Object-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Possible Applications

- 🚗 Traffic monitoring
- 👤 People detection
- 🚌 Vehicle detection
- 🐕 Animal detection
- 😷 Mask compliance monitoring
- 🏙️ Smart surveillance
- 🎥 Intelligent video analysis

## 🚀 Future Improvements

- Add more object classes
- Improve detection accuracy
- Optimize inference speed
- Add object counting
- Add real-time alerts
- Deploy the application online

## 👨‍💻 Author

**Muhammad Zia**

Software Engineering | Artificial Intelligence | Computer Vision | Python

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.