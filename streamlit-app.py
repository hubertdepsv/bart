import streamlit as st
from transformers import pipeline


def generate_header():
    st.title("Text summarization")


def summarize():
    text = st.text_area(label="Enter text")
    submit = st.button("Summarize")

    if submit:
        # use facebook's pre-trained model in case you don't have our model's weights locally
        st.subheader("Summary")
        with st.spinner(text="This might take a while..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            answer = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.write(answer[0]["summary_text"])
        st.text(
            f'The summary includes {len(answer[0]["summary_text"].split())} tokens while the input text has {len(text.split())} tokens.'
        )


if __name__ == "__main__":
    generate_header()
    summarize()
