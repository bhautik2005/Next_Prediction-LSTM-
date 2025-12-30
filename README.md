# 🔮 Next Word Predictor  
### LSTM-based Deep Learning NLP Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![NLP](https://img.shields.io/badge/NLP-LSTM-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

**Next Word Predictor** is an end-to-end **Natural Language Processing (NLP)** web application that predicts the **most probable next word** in a sentence using a **Long Short-Term Memory (LSTM)** neural network.

This project demonstrates **real-world deep learning deployment**, combining:
- Sequential text modeling  
- Trained LSTM networks  
- Flask-based backend serving  
- Interactive web-based UI  

After experimenting with **3 different LSTM architectures across 2 datasets**, the final selected model achieved a **validation accuracy of ~72%**, making it both reliable and production-ready for educational and prototype use cases.

---

## 🎯 Problem Statement

Language is sequential by nature. Traditional machine learning models fail to capture long-term dependencies in text.  
This project solves that by leveraging **LSTM networks**, which are specifically designed to remember context across long sequences.

**Goal:**  
> Given a partial sentence, predict the most contextually relevant next word.

---

## ✨ Key Features

- ✅ **Context-Aware Predictions** using LSTM  
- ⚡ **Real-Time Inference** through Flask API  
- 🧠 **Trained Deep Learning Model** (Saved & Reloaded)  
- 🔄 **Consistent Preprocessing** using saved Tokenizer & padding length  
- 🎨 **Clean Web Interface** using HTML & CSS  
- 📦 **Production-Ready Structure** with reusable artifacts  

---

## 🛠️ Tech Stack

### 🔹 Machine Learning & NLP
- **TensorFlow / Keras** – Model building, training & inference  
- **LSTM (RNN)** – Sequential language modeling  
- **NumPy** – Numerical computations  
- **Pickle** – Serialization of tokenizer and metadata  

### 🔹 Backend & Web
- **Flask** – Lightweight backend server  
- **HTML5 & CSS3** – Frontend interface  

---

## 🧠 Model Architecture & Approach

### 1️⃣ Text Preprocessing Pipeline
- **Tokenization**  
  Converts words into integer indices using a trained `Tokenizer`.
- **Sequence Creation**  
  Builds n-gram sequences to learn sentence progression.
- **Padding**  
  Uses a fixed `max_len_x` to normalize input length.

### 2️⃣ LSTM Neural Network
- Embedding Layer → LSTM Layer(s) → Dense Softmax Output  
- Designed to:
  - Capture long-term dependencies  
  - Handle variable-length text  
  - Avoid vanishing gradient issues common in vanilla RNNs  

### 3️⃣ Model Experimentation
| Model Version | Dataset | Result |
|--------------|--------|--------|
| Model 1 | Dataset A | Underfitting |
| Model 2 | Dataset B | Overfitting |
| **Model 3 (Final)** | Combined | **~72% Accuracy** ✅ |

---

## 📂 Project Structure

```bash
Next-Word-Predictor/
│
├── Data_set/
│   ├── processed_quotes_cleaned.csv   # Cleaned & preprocessed dataset
│   └── quote_dataset.csv              # Original raw text dataset
│
├── model/
│   ├── lstm_model.h5                  # Trained LSTM model (version 1)
│   ├── lstm_model2.h5                 # Trained LSTM model (version 2)
│   ├── lstm_model3.h5                 # Best-performing LSTM model (final)
│   ├── max_len_X.pkl                  # Serialized max sequence length
│   └── tokenizer.pkl                  # Serialized tokenizer object
│
├── static/
│   └── style.css                      # Frontend UI styling
│
├── templates/
│   └── index.html                     # Web application interface
│
├── app.py                             # Flask application entry point
├── requirements.txt                  # Python dependencies
└── README.md                          # Project documentation


⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/next-word-predictor.git
cd next-word-predictor

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py

5️⃣ Access the Web App

Open your browser and go to:

http://127.0.0.1:5000/

📦 Major Dependencies

flask

tensorflow

numpy

pickle-mixin

(See requirements.txt for full list)

📸 Screenshots

📌 Add screenshots of the UI here to improve visual appeal and recruiter impact.

🚀 Future Enhancements

🔮 Top-k / Top-n word predictions

📈 Beam search for better sentence generation

🧠 Transformer-based model (GPT-style)

🌍 Multi-language support

☁️ Cloud deployment (AWS / Render / Hugging Face Spaces)

🧑‍💻 Author

Bhautik Gondaliya
Aspiring Data Scientist | Machine Learning & NLP Enthusiast

This project reflects hands-on experience in Deep Learning, NLP pipelines, Flask deployment, and model lifecycle management.

⭐ Acknowledgments

TensorFlow & Keras Documentation

NLP research & sequence modeling concepts
