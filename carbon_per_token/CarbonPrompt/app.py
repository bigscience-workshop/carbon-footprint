import streamlit as st
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
MEASURE_INTERVAL = 1

st.title("CarbonPrompt")
st.text('CarbonPrompt is a tool by BigScience to help you measure the carbon footprint of your language model inputs')

model_checkpoint = st.text_input("Model Checkpoint (from Huggingface Hub)", key="model_checkpoint")

model_prompt = st.text_area("Model Input ", key="model_input")
run = st.button("Run", key="run_inference")

use_codecarbon = st.checkbox('Use CodeCarbon?')
# Used to display any warnings to tue user TODO: Does streamlit offer a notification system for this instead?
# TODO: It does: https://docs.streamlit.io/library/api-reference/status/st.error
model_warnings = ''
st.write(model_warnings)

# TODO: Display a bar chart with a running total for the session
# chart_data = pd.DataFrame(
#      np.random.randn(50, 3),
#      columns=["a", "b", "c"])

# st.bar_chart(chart_data)

if run:

    # TODO: We need to support tensorflow & jax here as well
    # It is important we know the device we're running.
    isGpu = torch.cuda.is_available()
    device = "cuda:0" if isGpu else "cpu"

    # Catch any potential warnings or error before running inference
    if model_checkpoint == '':
        model_warnings = "Please input a model checkpoint you would like to use."
    elif model_prompt == '':
        model_warnings = "Please input a prompt you would like to pass to the model."
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModel.from_pretrained(model_checkpoint)

            if use_codecarbon:
                from codecarbon import EmissionsTracker
                tracker = EmissionsTracker(project_name=f"{device}_{model_checkpoint}", measure_power_secs=MEASURE_INTERVAL)
                tracker.start()

            inputs = tokenizer(model_prompt)
            outputs = model.generate(model_prompt)

            if use_codecarbon:
                tracker.stop()

            # TODO: Display the results to the user


        except Exception:
            # TODO: Display the error to the user
            model_warnings = "Error when runing inference."
    
