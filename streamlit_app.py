import streamlit as st


st.title("Clustering of Retail shop customers.")
selected_type = st.radio("Choose data type:", ('Trimmed', 'Non-Trimmed'))
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
