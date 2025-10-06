# 🐦 LoveBirdID
### *A Hybrid Random Forest–HMM Approach for Automated Sex Classification of Lovebirds Based on Vocalization and Body Mass*

**LoveBirdID** is a Computer Engineering thesis project that automates the identification of lovebird (Agapornis spp.) gender using machine learning.  
It integrates **Random Forest (RF)** for analyzing biometric data such as *body mass*, and **Hidden Markov Models (HMM)** for processing *vocalization patterns*.  
This hybrid model provides a **non-invasive**, **cost-efficient**, and **accurate** alternative to traditional sexing methods like DNA testing and vent inspection.

---

## 🎯 Objectives
- To develop a hybrid RF–HMM system that classifies lovebird gender based on vocalization and body mass.  
- To evaluate the system’s performance in terms of accuracy, efficiency, and reliability.  
- To create a functional prototype for real-time or near-real-time classification.  
- To assess software performance using **ISO/IEC 25010** standards.

---

## 🧠 Features
- 🎙️ **Audio-Based Classification** — analyzes chirps and calls through MFCC and HMM.  
- ⚖️ **Body Mass Analysis** — uses Random Forest for biometric-based classification.  
- 🔄 **Hybrid Model Integration** — decision-level fusion of both algorithms for improved accuracy.  
- 💻 **User Interface** — web dashboard for visualization and result monitoring.  
- 📊 **Performance Metrics** — includes accuracy, precision, recall, F1-score, and confusion matrix.  

---

## ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| **Programming Language** | Python |
| **ML Libraries** | Scikit-learn, hmmlearn, TensorFlow, Keras |
| **Signal Processing** | Librosa, NumPy, SciPy |
| **Web Framework** | Flask, HTML, CSS, Bootstrap |
| **Database (Optional)** | SQLite / MySQL |
| **Hardware** | Raspberry Pi 4, HX711 Load Cell, INMP411 Microphone |

---

## 🧩 System Architecture
```
Audio Input ─┐
              │→ [Feature Extraction (MFCC)] → [Hidden Markov Model]
Body Mass ───┘
              → [Random Forest Classifier]
              → [Decision Fusion Layer]
              → [Final Gender Classification Output]
```

---

## 📂 Folder Structure
```
LoveBirdID/
│
├── dataset/                 # Lovebird recordings and biometric data
├── model/                   # Trained RF and HMM models (.pkl / .h5)
├── src/                     # Core scripts (training, preprocessing, fusion)
│   ├── train_rf.py
│   ├── train_hmm.py
│   ├── fusion_predict.py
│   └── utils/
│
├── webapp/                  # Flask web app and interface
│   ├── templates/
│   ├── static/
│   └── app.py
│
├── notebooks/               # Jupyter notebooks for testing and analysis
├── requirements.txt
└── README.md
```

---

## 🔧 Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/LoveBirdID.git
   cd LoveBirdID
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python webapp/app.py
   ```

4. **Access in Browser**
   ```
   http://localhost:5000
   ```

---

## 🧪 Model Training

### 🎙️ Train the HMM (Vocalization)
```bash
python src/train_hmm.py
```

### ⚖️ Train the Random Forest (Body Mass)
```bash
python src/train_rf.py
```

### 🔄 Combine and Fuse Predictions
```bash
python src/fusion_predict.py
```

---

## 📊 Evaluation Metrics
| Metric | Description |
|---------|-------------|
| **Accuracy** | Correct predictions over all samples |
| **Precision** | Proportion of true positive predictions |
| **Recall (Sensitivity)** | Model’s ability to find all positive samples |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **AUC-ROC** | Measures how well the model separates classes |

---

## 🧪 Hardware Components
| Component | Description |
|------------|-------------|
| Raspberry Pi 4 | Core processing unit |
| HX711 + Load Cell | Measures body mass |
| INMP411 Microphone | Captures bird vocalizations |
| 12V Battery | Power supply |
| Glass Enclosure | Acoustic isolation chamber |

---

## 👨‍💻 Developers

**Colegio de Muntinlupa — Computer Engineering Department**  
Bachelor of Science in Computer Engineering (A.Y. 2024–2025)

| Name | Role |
|------|------|
| **Ghyne Clarence F. Miguel** | ML Developer / Data Specialist |
| **Martin Joe V. Vinuya** | Hardware Integration / System Engineer |
| **Jan Christian D. Wagan** | Web Developer / UI-UX Designer |

**Thesis Adviser:** *Engr. Ricrey E. Marquez, PCpE*  
**Panel Chair:** *Engr. Orlando M. Lingo, PCpE*  
**Dean:** *Engr. Robert James Y. Lomotan, MSME*  
**College President:** *Dr. Teresita C. Fortuna, CESO III*

---

## 📜 License

```
MIT License (Academic Use Only)

Copyright (c) 2025 Colegio de Muntinlupa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this project and associated documentation files (the "Software"), to use,
copy, modify, merge, and distribute copies of the Software for educational and
research purposes only, subject to the following conditions:

1. The Software shall not be used for commercial purposes.
2. Proper credit must be given to the original authors and institution.
3. This copyright notice and permission notice shall be included in all copies.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
```

---

## 💬 Acknowledgements
Special thanks to:
- **Engr. Ricrey E. Marquez**, Thesis Adviser  
- **Engr. Orlando M. Lingo**, Panel Chair  
- **Engr. Tristan Jay Calaguas** and **Engr. Darius Añonuevo**, Panel Members  
- **Engr. Robert James Lomotan**, College Dean  
- **Dr. Teresita C. Fortuna**, College President  
- The **Computer Engineering Department**, Colegio de Muntinlupa  

---

> 🐥 *“Empowering avian science through machine intelligence.”*
