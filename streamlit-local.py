import streamlit as st
from transformers import pipeline
from summarizer import summarize_a_text


def generate_header():
    st.title("Text summarization")


def choose_model():
    option = st.selectbox(
        "Select a model",
        ["Our model", "Facebook pre-trained model"],
    )
    return option


def summarize(option):
    text = st.text_area(label="Enter text")
    submit = st.button("Summarize")
    if submit:
        if option == "Our model":
            answer = summarize_a_text(
                text,
                path="../model_files",
                source_len=1024,
                max_length_pred=128,
                min_length_pred=30,
            )
            st.write(answer)
            st.text(
                f"The summary includes {len(answer.split())} tokens while the input text has {len(text.split())} tokens."
            )

        elif option == "Facebook pre-trained model":
            # use facebook's pre-trained model in case you don't have our model's weights locally
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            answer = summarizer(text, max_length=130, min_length=30, do_sample=False)
            st.write(answer[0]["summary_text"])
            st.text(
                f'The summary includes {len(answer[0]["summary_text"].split())} tokens while the input text has {len(text.split())} tokens.'
            )


if __name__ == "__main__":
    generate_header()
    option = choose_model()
    summarize(option)
