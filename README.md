# 🔍 Deep Learning-Based Visual Quality Inspection and Defect Detection for PCBs

A **deep learning-powered PCB (Printed Circuit Board) defect detection system** built with **YOLOv8** for automated quality inspection.  
The system detects common PCB defects such as **missing holes, mouse bites, shorts, and soldering issues**, and integrates with a **Streamlit interface** for single, batch, and real-time analysis.  
It supports **human-in-the-loop verification**, **report generation**, and **Firebase storage** for traceability.  

---

## 🚀 Features

- 📷 **Single PCB Analysis** – Upload an image, view defects with bounding boxes.  
- 📂 **Batch Processing** – Upload multiple images, generate CSV/JSON summaries.  
- 🎥 **Real-Time Detection** – Live inspection using webcam/iPhone camera.  
- 👨‍💻 **Human-in-the-Loop** – Accept, reject, or relabel detections.  
- 📝 **Report Generation** – Export annotated images and PDF defect reports.  
- ☁️ **Database Integration** – Store inspection results in Firebase.  
- 🔄 **Model Retraining Pipeline** – Retrain YOLOv8 with corrected labels.  

---

## 🛠️ Tech Stack

- **Language**: Python 3.10  
- **Deep Learning**: PyTorch, Ultralytics YOLOv8  
- **UI/Frontend**: Streamlit  
- **Backend (optional)**: Flask APIs  
- **Database**: Firebase (Realtime DB / Firestore)  
- **Deployment**: Docker  
- **Annotation Tool**: LabelImg  

---

## 📂 Project Structure

pcb-defect-detection/
├── data/                     # Dataset (train/val/test)
├── notebooks/                # Jupyter notebooks for training & evaluation
│   ├── preprocessing.ipynb
│   ├── train_yolo.ipynb
│   └── evaluate_model.ipynb
├── src/                      # Source code
│   ├── app.py                # Streamlit main app
│   ├── firebase_utils.py     # Firebase integration
│   ├── pdf_report.py         # Report generation
│   └── utils/                # Helper functions
├── requirements.txt          # Python dependencies
├── Dockerfile                # Containerized deployment
├── README.md                 # Project documentation
└── LICENSE                   # License file

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/pcb-defect-detection.git
cd pcb-defect-detection

### 2️⃣ Create Virtual Environment

python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate


### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Setup Firebase
- Create a Firebase project via [Firebase Console](https://console.firebase.google.com/).  
- Enable **Realtime Database / Firestore**.  
- Download your `firebase_config.json` and place it in `src/`.  

---

## ▶️ Running the Application

Launch the **Streamlit app**:

streamlit run main.py


Open [http://localhost:8501](http://localhost:8501) in your browser.  

- **Single PCB Analysis** → Upload one image for detection  
- **Batch Analysis** → Upload multiple images and generate reports  
- **Real-Time Analysis** → Use webcam/iPhone camera for live inspection  

---

## 📊 Model Training & Evaluation

### Train on custom dataset

yolo detect train data=data/pcb.yaml model=yolov8n.pt epochs=100 imgsz=640


### Evaluate trained model

yolo detect val model=runs/detect/train/weights/best.pt data=data/pcb.yaml


The following metrics are generated:  
- **Precision, Recall, F1-score**  
- **mAP@0.5**  
- **mAP@[0.5:0.95]**  

---

## 📑 Reports

- **PDF reports**: Defect summary, confidence scores, bounding boxes, QA recommendations.  
- **Batch mode**: Generates CSV/JSON summaries for integration with production logs.  

---

## 📌 Future Roadmap

- 📈 Expand dataset with rare defect categories.  
- 🔗 Integrate with automated assembly line systems.  
- 🏷️ Add severity classification (minor vs critical defects).  
- ⚡ Deploy on **edge devices** (TensorRT/ONNX optimization).  
- 🧠 Introduce **Explainable AI (XAI)** for transparent decision-making.  

---

## 👨‍💻 Author

** L.M Madusha Nadiranga** – Final Year Project, ICBT Campus  
** Supervisor**: Dr. Gayan Galhena  

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

✨ _Built with Deep Learning to ensure reliable PCB quality inspection._  
