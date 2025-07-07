import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from textblob import TextBlob
from scipy.sparse import hstack

# -----------------
# Text Cleaning & Feature Functions
# -----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def detect_emotion(text):
    text = text.lower()
    if 'happy' in text or 'joy' in text:
        return 'joy'
    elif 'sad' in text or 'cry' in text:
        return 'sadness'
    elif 'angry' in text or 'fight' in text:
        return 'anger'
    elif 'fear' in text or 'scared' in text:
        return 'fear'
    return 'neutral'

# -----------------
# Load Saved Components
# -----------------
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('genre_model.pkl', 'rb') as f:
    model = pickle.load(f)

# -----------------
# App UI
# -----------------
st.set_page_config(page_title="🎬 Movie Genre Prediction App", page_icon="🎬", layout="wide")

st.title("🎬 Movie Genre Prediction App")

# Session History
if "history" not in st.session_state:
    st.session_state["history"] = []

# User Input
desc_input = st.text_area("📝 Enter Movie Description:")

# Predict Button
if st.button("🎯 Predict Genre"):
    if desc_input.strip() == "":
        st.warning("⚠️ Please enter a description.")
    else:
        clean_desc = clean_text(desc_input)
        sentiment = get_sentiment(clean_desc)
        emotion = detect_emotion(clean_desc)

        # Sentiment & Emotion Side by Side
        col1, col2 = st.columns(2)
        col1.info(f"🧭 Sentiment Polarity: {sentiment:.2f}")
        col2.info(f"🎭 Detected Emotion: {emotion.capitalize()}")

        # Vectorize Description
        X_text = tfidf_vectorizer.transform([clean_desc])

        # Sentiment & Emotion Features
        sentiment_feat = np.array([[sentiment]])
        emotion_map = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'neutral': 4}
        emotion_encoded = np.zeros((1, 5))
        emotion_encoded[0, emotion_map.get(emotion, 4)] = 1

        # Combine Features
        X_final = hstack([X_text, sentiment_feat, emotion_encoded])

        # Predict Probabilities
        proba = model.predict_proba(X_final)[0]
        top3_idx = proba.argsort()[-3:][::-1]

        # Extended Emoji Feedback
        genre_emojis = {
            'comedy': '😂', 'horror': '👻', 'drama': '🎭', 'thriller': '🔪',
            'romance': '❤️', 'documentary': '🎥', 'crime': '🚔', 'adult': '🔞',
            'reality-tv': '📺', 'animation': '🐭', 'fantasy': '🧙', 'musical': '🎶',
            'war': '⚔️', 'sci-fi': '🚀', 'mystery': '🕵️', 'western': '🤠',
            'history': '🏛️', 'biography': '📚', 'sport': '🏅', 'family': '👨‍👩‍👧'
        }
        top_genre = label_encoder.inverse_transform([top3_idx[0]])[0]
        emoji = genre_emojis.get(top_genre.lower(), '🎬')

        st.markdown(f"""
        <div style='
            background-color: #f3e5f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            text-align: center;
            margin: 20px 0;
        '>
            <h2 style='color: #28a745;'>🎯 Predicted Genre: {top_genre} {emoji}</h2>
        </div>
        """, unsafe_allow_html=True)


        # Top 3 Genres in Expander
        with st.expander("⭐ See Top 3 Predicted Genres with Confidence"):
            for idx in top3_idx:
                genre = label_encoder.inverse_transform([idx])[0]
                confidence = proba[idx] * 100
                genre_emoji = genre_emojis.get(genre.lower(), '🎬')
                st.write(f"- **{genre}** : {confidence:.2f}%")

        # Probability Bar Chart in Expander with Fresh Colors
        with st.expander("📊 See Full Probability Distribution"):
            fig, ax = plt.subplots(figsize=(6,4))
            palette = sns.color_palette("viridis", len(label_encoder.classes_))
            colors = [palette[i] for i in range(len(label_encoder.classes_))]
            ax.barh(label_encoder.classes_, proba * 100, color=colors)
            ax.set_xlabel("Probability (%)")
            ax.set_ylabel("Genre", fontsize=12)
            ax.set_title("Genre Probability Distribution")
            ax.tick_params(axis='y', labelsize=10)
            ax.invert_yaxis()
            st.pyplot(fig)

        # Save to History
        st.session_state.history.append({
            "Description": desc_input,
            "Top Genre": top_genre,
            "Confidence": f"{proba[top3_idx[0]]*100:.2f}%"
        })

# Sidebar for History Download
with st.sidebar:
    st.markdown("### 📜 Prediction History 📂")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        with st.expander("📝 View Recent Predictions"):
            st.dataframe(history_df.tail(3), use_container_width=True)

        st.download_button(
            label="💾 Download Full History as CSV",
            data=history_df.to_csv(index=False).encode('utf-8'),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions yet. Start by entering a movie description! 🎬")

