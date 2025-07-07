# 🎬 Movie Genre Classification with Sentiment & Emotion Features

This project is a **Machine Learning-powered web app** that predicts the **genre of a movie** based on its description. It uses **TF-IDF vectorization**, **sentiment analysis**, **basic emotion detection**, and a trained classification model to make genre predictions. The app is built using **Streamlit** for an interactive, user-friendly experience.

---

## 🚀 Features

✅ Predicts movie genre based on description  
✅ Displays top 3 most probable genres with confidence scores  
✅ Includes sentiment polarity detection (using TextBlob)  
✅ Includes basic emotion detection (keyword-based)  
✅ Shows full probability distribution for all genres  
✅ Interactive, responsive Streamlit UI with emoji-enhanced design  
✅ Downloadable prediction history as CSV  

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression (or Random Forest)  
- TextBlob (for sentiment analysis)  
- Streamlit (frontend)  
- Matplotlib & Seaborn (visualizations)  
- Numpy & Pandas (data handling)  

---

## 📦 Project Structure

```
MOVIE_GENRE_CLASSIFICATION/
├── Genre Classification Dataset/
│ ├── description.txt               # Dataset description
│ ├── test_data_solution.txt        # Ground truth for test data
│ ├── test_data.txt                 # Unlabeled test data
│ └── train_data.txt                # Training dataset
├── app.ipynb                       # EDA & model building
├── app.py                          # Final Streamlit web app
├── genre_model.pkl                 # Trained ML model
├── label_encoder.pkl               # Saved Label Encoder
├── tfidf_vectorizer.pkl            # Saved TF-IDF vectorizer
├── requirements.txt                # Project dependencies
└── README.md
```
---

## 💻 How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Movie-Genre-Classification.git
cd Movie-Genre-Classification
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
(*Generate requirements with `pip freeze > requirements.txt` if needed.*)

3. **Run the app**
```bash
streamlit run app.py
```

---

## 💡 Unique Features

- Predict movie genres based on descriptions  
- Top 3 genre predictions with confidence scores  
- Sentiment polarity detection using TextBlob  
- Basic emotion detection (joy, sadness, anger, fear, neutral)  
- Visual probability distribution chart  
- Downloadable prediction history

---

## 🧑‍💻 Developed For

CodSoft Internship - Machine Learning Intern  
Project: Movie Genre Classification with Sentiment & Emotion Features

---

## 📬 Contact

Feel free to connect:  
[LinkedIn]([https://www.linkedin.com/in/mrunal-gaikwad](https://www.linkedin.com/in/mrunal-gaikwad-328273300)) | [GitHub](https://github.com/mrunalgaikwad2364/Movie_Genre_Classification.git)  
