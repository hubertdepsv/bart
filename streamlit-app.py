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
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    answer = summarizer(text, max_length=130, min_length=30, do_sample=False)
    st.write(answer[0]["answer"])
