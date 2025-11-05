# ✈️ Aircraft Maintenance Prediction (LSTM Model)

An **AI-powered Aerodynamic Maintenance Prediction System** that uses a **Long Short-Term Memory (LSTM)** deep learning model to predict whether an aircraft engine will **fail within the next 30 flight cycles** based on real engine sensor data.  

This project features an interactive **Streamlit web app** where users can upload sensor data (CSV) and receive **real-time predictions** about the aircraft’s engine health.

---

## 🧠 Project Overview

Aircraft maintenance is one of the most critical aspects of aviation safety and operational efficiency.  
This project leverages **time-series sensor data** and **LSTM neural networks** to anticipate potential engine failures before they happen, enabling **predictive maintenance** and reducing downtime.

The system:
- Processes sequential sensor data
- Predicts engine failure probability
- Classifies engine status as *Healthy* or *Failure Soon*

---

## 🚀 Features

- ✅ **Interactive Streamlit Interface**
- 🧠 **Deep Learning Model (LSTM)** trained on sequential engine sensor data
- ⚙️ **Automatic File Detection** for model and scaler
- 📂 **Supports Custom Model Uploads** (`.keras`, `.h5`, `.pkl`)
- 🔍 **Input Validation** (min. 50×25 CSV shape)
- 📊 **Real-Time Prediction Dashboard**
- 🔒 **Safe Prediction Handling** with robust error catching

---

## 🧩 Tech Stack

| Category | Technology |
|-----------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Model Serialization** | Joblib |
| **Deployment** | Streamlit Cloud / Local Execution |

---

## 🗂️ Project Structure

```bash
├── app.py                        # Streamlit web app
├── models/
│   ├── lstm_model.keras          # Pre-trained LSTM model
│   └── scaler.pkl                # Scaler for normalization
├── sequence_cols.json            # Sequence column mapping
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
