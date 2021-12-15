import streamlit as st


@st.cache

option = st.selectbox(
    "Select an Option",
    ["Text Summarization", ]
)



if __name__ == "__main__":
    # df = get_data()
    # generate_dashboard(df)
