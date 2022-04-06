import streamlit as st
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
MEASURE_INTERVAL = 1

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_tokenizer(model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(model_checkpoint: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model

def run(tokenizer, model):
    # Variables 
    total_tokens = 0

    st.title("CarbonPrompt")
    st.text('CarbonPrompt is a tool by BigScience to help you measure the carbon footprint of your language model inputs')

    model_checkpoint = st.text_input("Model Checkpoint (from Huggingface Hub)", key="model_checkpoint")

    model_prompt = st.text_area("Model Input ", key="model_input")
    run = st.button("Run", key="run_inference")

    use_codecarbon = st.checkbox('Use CodeCarbon?')

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

        # Clear the total tokens variable for this run
        total_tokens = 0

        # Catch any potential warnings or error before running inference
        if model_prompt == '':
            print("Please input a prompt you would like to pass to the model.")
        else:
            try:
                print(f"use_codecarbon is: {use_codecarbon}")
                if use_codecarbon:
                    from codecarbon import EmissionsTracker
                    tracker = EmissionsTracker(project_name=f"{device}_{model_checkpoint}")
                    tracker.start()
                    print("starting codecarbon")

                inputs = tokenizer.encode(model_prompt, return_tensors="pt").to(device)
                print(f"inputs: {inputs}")
                # TODO: This length is wrong bc of the tensor shape
                total_tokens += len(inputs)
                outputs = model.generate(inputs)
                print(f"Outputs: {outputs}")

                if use_codecarbon:
                    tracker.stop()
                    print("stopping codecarbon")

                # TODO: Display the results to the user
                print(f"Total- tokens: {total_tokens}")


            except Exception as e:
                # TODO: Display the error to the user
                print(f"Error when runing inference: {e}")

if __name__=='__main__':
    # TODO: Pass this as a cli argument
    tokenizer = load_tokenizer('bigscience/T0_3B')
    model = load_model('bigscience/T0_3B')
    run(tokenizer, model)
    
