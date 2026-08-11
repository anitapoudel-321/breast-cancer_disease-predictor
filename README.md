🧠 Breast Cancer Diagnosis Predictor (SVM)

A machine learning web application that predicts whether a tumor is Benign or Malignant using a trained Support Vector Machine (SVM) model.
The app is built using Python and deployed with Streamlit.

🚀 Live Demo

👉 Try the app here:

https://anitapoudel-321-cancer-disease-app-v2tnpx.streamlit.app


 Features
Predicts breast cancer diagnosis (Benign / Malignant)

Uses SVM (Support Vector Machine) model

Interactive UI with real-time input sliders

Displays prediction confidence

Clean and user-friendly interface

🛠️ Tech Stack
Python 
Streamlit
Scikit-learn (SVM Model)
Pandas & NumPy
Joblib (for model saving/loading)


📂 Project Structure
├── app.py                # Main Streamlit app

├── svm_model.pkl        # Trained SVM model

├── scaler.pkl           # Data scaler

├── data.csv             # Dataset

├── breast.png           # UI image
├── requirements.txt     # Dependencies
└── svm.ipynb            # Model training notebook
