# 🐦 LoveBirdID
### *A Hybrid Random Forest–HMM Approach for Automated Sex Classification of Lovebirds Based on Vocalization and Body Mass*

LoveBirdID is a thesis project designed to **automate the identification of lovebird (Agapornis spp.) gender** using **machine learning**.  
The system combines **Random Forest (RF)** for analyzing biometric data such as *body mass*, and **Hidden Markov Models (HMM)** for processing *vocalization patterns*.  
This hybrid AI approach provides a **non-invasive, cost-effective, and accurate** alternative to traditional sexing methods like DNA testing or vent inspection.

---

## 🎯 Objectives
- Develop an intelligent hybrid RF–HMM system for classifying the sex of lovebirds using biometric and vocal inputs.  
- Evaluate model performance in terms of **accuracy, efficiency, and reliability**.  
- Implement a functional prototype capable of **real-time or near-real-time** processing.  
- Assess software quality using **ISO/IEC 25010** standards.

---

## 🧠 Features
- 🎙️ **Audio-Based Classification** — analyzes chirps and calls using MFCC feature extraction and HMM  
- ⚖️ **Body Mass Analysis** — applies Random Forest algorithm to biometric weight data  
- 🔄 **Hybrid Model Integration** — decision-level fusion of RF and HMM outputs for improved precision  
- 💻 **Web-Based Interface** — displays predictions and performance metrics  
- 📊 **Model Evaluation** — includes accuracy, precision, recall, and F1-score visualization  

---

## ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| **Programming Language** | Python |
| **Machine Learning Libraries** | Scikit-learn, hmmlearn, TensorFlow/Keras |
| **Signal Processing** | Librosa, NumPy, SciPy |
| **Interface / Dashboard** | Flask, HTML, CSS, Bootstrap |
| **Database (optional)** | SQLite / MySQL |
| **Hardware** | Raspberry Pi 4, HX711 Load Cell, INMP411 Microphone |

---

## 🧩 System Architecture
