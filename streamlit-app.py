import streamlit as st
from transformers import pipeline
from summarizer import summarize_a_text

option = st.selectbox(
    "Select an Option",
    [
        "Text Summarization",
    ],
)

text = st.text_area(label="Enter text")
if text:
    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    #answer = summarizer(text, max_length=130, min_length=30, do_sample=False)
    #st.write(answer[0]["summary_text"])

    answer = summarize_a_text(text, path='model_files', source_len=1024, 
        max_length_pred=128, min_length_pred=30)
    st.write(answer)
