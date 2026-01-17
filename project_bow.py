import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- DOWNLOADS ----------------
nltk.download("punkt")
nltk.download("stopwords")

nlp = spacy.load("en_core_web_sm")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NLP Processing App",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🧠 NLP Processing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization, and Bag of Words")

# ---------------- INPUT ----------------
text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Sayantan is HOD of HIT and loves NLP"
)

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words"
    ]
)

# ---------------- PROCESS BUTTON ----------------
if st.button("🚀 Process Text"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    
    # ---------------- TOKENIZATION ----------------
    elif option == "Tokenization":
        st.subheader("🔹 Tokenization Output")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            st.write(sent_tokenize(text))

        with col2:
            st.markdown("### Word Tokenization")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("### Character Tokenization")
            st.write(list(text))

    # ---------------- TEXT CLEANING ----------------
    elif option == "Text Cleaning":
        st.subheader("🔹 Text Cleaning Output")

        text_lower = text.lower()
        cleaned_text = "".join(
            ch for ch in text_lower
            if ch not in string.punctuation and not ch.isdigit()
        )

        doc = nlp(cleaned_text)
        final_words = [token.text for token in doc if not token.is_stop]

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    # ---------------- STEMMING ----------------
    elif option == "Stemming":
        st.subheader("🔹 Stemming Output")

        words = word_tokenize(text)

        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": [porter.stem(w) for w in words],
            "Lancaster Stemmer": [lancaster.stem(w) for w in words]
        })

        st.dataframe(df, use_container_width=True)

    # ---------------- LEMMATIZATION ----------------
    elif option == "Lemmatization":
        st.subheader("🔹 Lemmatization using spaCy")

        doc = nlp(text)
        df = pd.DataFrame(
            [(token.text, token.pos_, token.lemma_) for token in doc],
            columns=["Word", "POS", "Lemma"]
        )

        st.dataframe(df, use_container_width=True)

    # ---------------- BAG OF WORDS ----------------
    elif option == "Bag of Words":
        st.subheader("🔹 Bag of Words Representation")

        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform([text])

        df = pd.DataFrame({
            "Word": vectorizer.get_feature_names_out(),
            "Frequency": X.toarray()[0]
        }).sort_values(by="Frequency", ascending=False)

        st.markdown("### BoW Frequency Table")
        st.dataframe(df, use_container_width=True)

        # -------- PIE CHART --------
        st.markdown("### Word Frequency Distribution (Top 10)")

        top_n = 10
        df_top = df.head(top_n)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            df_top["Frequency"],
            labels=df_top["Word"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")

        st.pyplot(fig)
