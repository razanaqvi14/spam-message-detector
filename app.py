import streamlit as st
import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

st.set_page_config(
    page_title="Spam Message Detector", page_icon="📧", layout="centered"
)

st.title("Spam Message Detector")

st.divider()

st.info(
    "- This is a spam message detector that utilizes a machine learning model to identify whether a given message is spam.\n\n- The model employed here is a Random Forest Classifier, which boasts an accuracy of 96%. For verification, the model's accuracy is provided at the end of this webpage."
)


@st.cache_resource
def download_nltk_data():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


download_nltk_data()


@st.cache_resource
def load_models():
    word2vec_model = Word2Vec.load("./models/hamvsspamword2vec.model")
    classifier_model = joblib.load("./models/hamvsspamclassifier.pkl")
    return word2vec_model, classifier_model


word2vecmodel, classifier = load_models()


def clean_text(text):
    text = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    text = re.sub(r"(?<=\w)\\'(?=\w)", "", text)
    text = re.sub(r"(?<=\w)\'(?=\w)", "", text)
    text = re.sub(r"&lt;#&gt;", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text.strip()


def preprocess_text(text):
    text = " ".join(
        [
            WordNetLemmatizer().lemmatize(token)
            for token in word_tokenize(text)
            if token not in set(stopwords.words("english"))
        ]
    )
    return text.strip()


def average_word2vec(doc):
    return np.mean(
        [
            word2vecmodel.wv[word]
            for word in doc
            if word in word2vecmodel.wv.index_to_key
        ],
        axis=0,
    )


def average_word2vec(doc):
    return np.mean(
        [
            word2vecmodel.wv[word]
            for word in doc
            if word in word2vecmodel.wv.index_to_key
        ],
        axis=0,
    )


def predict(text):
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)
    tokenized_text = word_tokenize(preprocessed_text)
    mean_vector = average_word2vec(tokenized_text)
    return classifier.predict([mean_vector])


message = st.text_area("Message", placeholder="Message")

if st.button("Detect", type="primary"):
    if not message.strip():
        st.warning("Please enter any message first before clicking the button.")
    else:
        prediction = predict(message)
        if prediction == 1:
            st.error("The message is SPAM")
        else:
            st.success("The message is NOT SPAM")

st.divider()
st.subheader("Model's Classification Report & Confusion Matrix")
st.image("./model's report.png")
st.info(
    "1 is for SPAM and 0 for NOT SPAM\n\nDon't worry if you aren't able to understand this just focus on the accuracy, a machine learning engineer would understand it ;)"
)
st.markdown(
    "<h5 style='text-align: center;'>made by NEXUS</h5>",
    unsafe_allow_html=True,
)
