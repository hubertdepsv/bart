import streamlit as st
from transformers import pipeline

option = st.selectbox(
    "Select an Option",
    [
        "Text Summarization",
    ],
)

text = st.text_area(label="Enter text")
if text:
    classifier = pipeline("summarization")
    answer = classifier(text, max_length=200, min_length=10)
    st.write(answer)
