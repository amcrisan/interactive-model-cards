### LIBRARIES ###
# # Data
import numpy as np
import pandas as pd
import json
from math import floor

# Robustness Gym and Analysis
import robustnessgym as rg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# App & Visualization
import streamlit as st
import altair as alt
from streamlit_vega_lite import altair_component

#utils
from interactive_model_cards import utils as ut
from interactive_model_cards import app_layout as al
from random import sample


### LOADING DATA ###
# model card data
@st.experimental_memo
def load_model_card():
    with open("./assets/data/text_explainer/model_card.json") as f:
        mc_text = json.load(f)
    return mc_text


# pre-computed robusntess gym dev bench
#@st.experimental_singleton
@st.cache(allow_output_mutation=True)
def load_data():
    # load dev bench
    devBench = rg.DevBench.load("./assets/data/rg/sst_db.devbench")
    return (devBench)

# load model
@st.experimental_singleton
def load_model():
    model = rg.HuggingfaceModel(
        "distilbert-base-uncased-finetuned-sst-2-english", is_classifier=True
    )
    return model


#@st.experimental_memo
def load_examples():
    with open("./assets/data/user_data/example_sentence.json") as f:
        examples = json.load(f)
    return examples


# loading the dataset
def load_basic():
    # load data
    devBench = load_data()
    # load model
    model = load_model()
    return devBench, model



if __name__ == "__main__":

    ### STREAMLIT APP CONGFIG ###
    st.set_page_config(layout="wide", page_title="Interactive Model Card")

    #import custom styling
    ut.init_style()

    ### LOAD DATA AND SESSION VARIABLES ###

    # ******* loading the mode and the data
    with st.spinner():
        sst_db, model = load_basic()

    #load example sentences
    sentence_examples = load_examples()

    # ******* session state variables
    if 'user_data' not in st.session_state:
        st.session_state['user_data'] = pd.DataFrame()
    if 'example_sent' not in st.session_state:
        st.session_state['example_sent'] = "I like you. I love you"
    if 'quant_ex' not in st.session_state:
        st.session_state['quant_ex'] = {
            'Overall Performance' : sst_db.metrics['model']
        }
    #if 'user_bench' not in st.session_state:
    #    st.session_state['user_bench'] = ut.new_bench()


    ### STREAMLIT APP LAYOUT###

    # ******* MODEL CARD PANEL ******* 
    st.sidebar.title("Interactive Model Card")
    # load model card data
    model_card = load_model_card()
    al.model_card_panel(model_card)

    # ******* QUANTITATIVE DATA PANEL ******* 
    al.quant_panel(sst_db)


    # ******* USER EXAMPLE DATA PANEL ******* 
    st.markdown("---")
    st.write("""<h1 style="font-size:20px;padding-top:0px;"> Additional Examples</h1>""", unsafe_allow_html=True)

    al.example_panel(sentence_examples,model)


                
