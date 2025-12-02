# 📰 Fake News Detection Dashboard  
A modern hybrid AI system for detecting fake news, built with Streamlit, HuggingFace Transformers, and a local ML fallback model.

<img width="1918" height="812" alt="image" src="https://github.com/user-attachments/assets/7337480f-dccf-4545-980e-f4139824b7e1" />

---

Try the app:
https://fake-news-prediction-brp3oqtbdsj5jd9pithszn.streamlit.app/

---

## 🚀 Overview  
The **Fake News Detection Dashboard** is an end-to-end NLP application that analyzes news headlines or short articles to determine whether they are **Real** or **Fake**, with multiple layers of interpretability and insights.

This project is designed as both a **useful misinformation analysis tool** and a **showcase of full-stack machine learning engineering**, featuring a polished UI, robust deployment design, and hybrid inference system.

---

## 🔥 Features

### **1. Hybrid AI Prediction System**
- **Primary Model:**  
  HuggingFace Transformer (`Fake-News-Bert-Detect`) via Router API  
- **Secondary Fallback Model:**  
  TF-IDF + Logistic Regression trained on `fake.csv`  
- Ensures predictions even if API is down  
- Works seamlessly on Streamlit Cloud

### **2. Rich NLP Insights**
For each headline/article, the app generates:

- ✔ Fake/Real Classification  
- ✔ Confidence Score & Progress Meter  
- ✔ Sentiment Analysis (VADER)  
- ✔ Topic Category Detection  
- ✔ Writing Complexity  
- ✔ Keyword Extraction  
- ✔ Misinformation Risk Assessment  

### **3. Clean & Engaging UI/UX**
- Modern dark gradient theme  
- Card-based layout  
- Auto-analysis for example headlines  
- Mobile-friendly  
- Stylish typography & colors  
- Professional-grade interface  

---

## 🧠 How It Works

### **Primary (Cloud Transformer Model)**
The app first sends the input text to:
https://router.huggingface.co/jy46604790/Fake-News-Bert-Detect


### **Fallback (Local ML Model)**
If the API fails or is rate-limited:

- A local TF-IDF vectorizer transforms the text  
- Logistic Regression predicts `Fake (0)` or `Real (1)`  

This ensures:
- Zero downtime  
- Works offline  
- Works without API keys  

---

## 📁 Project Structure

fake-news-prediction/
│
├── app.py # Main Streamlit application
├── fake.csv # Training dataset
├── requirements.txt # Python dependencies
├── README.md # Documentation
├── LICENSE # MIT License
└── assets/


---

## 📊 Dataset

The project uses a labeled fake news dataset structured as:

| id | title | author | text | label |
|----|--------|---------|--------|-------|
| 0  | ...    | ...     | ...    | 1 (Real) |
| 1  | ...    | ...     | ...    | 0 (Fake) |

---

## 🛠 Installation

### **Clone Repo**
```bash
git clone https://github.com/<your-username>/fake-news-prediction.git
cd fake-news-prediction
pip install -r requirements.txt
streamlit run app.py

```

🔑 Environment Variables

Create a .env or use system environment:
HF_TOKEN=your_huggingface_api_token
Without a token, the app automatically switches to local ML inference.

---

☁️ Deploy on Streamlit Cloud

Push repo to GitHub

Visit: https://streamlit.io/cloud

Select your repo

App runs automatically

No GPU required.

---

🧪 Technologies Used

Python 3.9+

Streamlit

HuggingFace Router API

Scikit-learn

TF-IDF Vectorizer

Logistic Regression

VADER Sentiment Analysis

Pandas, NumPy

---

🧬 Future Improvements

Fake news explanation using SHAP-like highlighting

Full article analysis

Support for multiple languages

Misinformation timeline visualization

Add a database for analysis history

---

🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss your ideas.

---

📜 License

This project is licensed under the MIT License — see LICENSE
 for details.
