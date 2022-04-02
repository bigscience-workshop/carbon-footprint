import streamlit as st
st.title("CarbonPrompt")
st.text('CarbonPrompt is a tool by BigScience to help you measure the carbon footprint of your language model inputs')


col1, col2 = st.columns(2)
with col1:
    st.text_input("Model Checkpoint (from Huggingface Hub)", key="model_checkpoint")

with col2:
    st.button("Download", key="download_checkpoint")



st.text_area("Model Input ", key="model_input")