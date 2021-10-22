### LIBRARIES ###
# # Data
import numpy as np
import pandas as pd
import json

# Robustness Gym and Analysis
import robustnessgym as rg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# App & Visualization
import streamlit as st
import altair as alt
from streamlit_vega_lite import altair_component


### LOADING DATA ###

# model card data
@st.experimental_memo
def load_model_card():
    with open("./assets/data/text_explainer/model_card.json") as f:
        mc_text = json.load(f)
    return mc_text


# robusntess gym from hugging face
@st.experimental_singleton
def load_data():
    # load dev bench
    devBench = rg.DevBench.load("./assets/data/rg/sst_db.devbench")
    return devBench


# load model
@st.experimental_singleton
def load_model():
    model = rg.HuggingfaceModel(
        "distilbert-base-uncased-finetuned-sst-2-english", is_classifier=True
    )
    return model


# loading the dataset
def load_basic():
    # load data
    devBench = load_data()
    # load model
    model = load_model()
    return devBench, model


### STREAMLIT APP CONGFIG ###
st.set_page_config(layout="wide", page_title="Interactive Model Card")

# page style with a slightly scary hack
st.write(
    """
    <style>
    /* Side Bar */
    .css-1outpf7 {
        background-color:rgb(246 240 240);
        width:40rem;
        padding:10px 10px 10px 10px;
    }
    /* Main Panel*/
    .css-18e3th9 {
        padding:10px 10px 10px 10px;
    }
    .css-1ubw6au:last-child{
        background-color:lightblue;
    }

    /* Model Panels : element-container */
    .element-container{
            border-style:none
    }

    div.row-widget.stRadio > div{flex-direction:row;}
    </style>
""",
    unsafe_allow_html=True,
)

### STREAMLIT APP LAYOUT###
# ******* side bar setup
st.sidebar.title("Interactive Model Card")

# load model card data
model_card = load_model_card()

with st.spinner():
    sst_db, model = load_basic()


# side panel
st.sidebar.markdown(f"**{model_card['model-details']['name']}**")
st.sidebar.write(f"{model_card['model-details']['short'][0]}")

with st.sidebar.expander("more details"):
    st.markdown(f"*{model_card['model-details']['short'][0]}")
    st.markdown(f"*{model_card['model-details']['short'][1]}")


# ******* quantaitive analysis
st.markdown("""**Quantaitive Analysis**""")
quant_lcol, quant_rcol = st.columns([6, 6])

with quant_lcol:
    st.write(sst_db.metrics)
with quant_rcol:
    st.write("Right Column")


# ******* Example Layout
with st.expander("Add your own examples", expanded=True):
    data_src = st.radio(
        "Select Example Source",
        ["Text Example", "From Training Data", "From Your Data"],
    )
    st.markdown("""---""")

    exp_lcol, exp_mid, exp_rcol = st.columns([5, 5, 2])

    if data_src == "From Training Data":
        exp_lcol.write("You training data")
        exp_mid.write("You training data 2")
        exp_rcol.write("You training data 3")

    elif data_src == "From Your Data":
        exp_lcol.write("Loading your own data")
        exp_mid.write("Loading your own data 2")
        exp_rcol.write("Loading your own data 3")
    else:
        user_text = exp_lcol.text_input(
            "Add your own example text", "I like you. I love you"
        )

        # adding user data to the data panel
        dp2 = rg.DataPanel({"sentence": [user_text], "label": [1]})
        dp2._identifier = f"User Input - {user_text}"

        # run prediction
        model.predict_batch(dp2, ["sentence"])
        dp2 = dp2.update(
            lambda x: model.predict_batch(x, ["sentence"]),
            batch_size=4,
            is_batched_fn=True,
            pbar=True,
        )

        sst_db.add_slices([dp2])

        exp_mid.write(sst_db.metrics)
