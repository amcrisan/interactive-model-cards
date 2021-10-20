### LIBRARIES ###
# App
import streamlit as st

# Data
import numpy as np
import pandas as pd
import json

#Visualization
import altair as alt
from streamlit_vega_lite import altair_component

### LOADING DATA ###
@st.experimental_memo
def load_model_card():
    with open('./data/text_explainer/model_card.json') as f:
        mc_text= json.load(f)
    return(mc_text)


### STREAMLIT APP CONGFIG ###
st.set_page_config(
    layout="wide",
    page_title = "Interactive Model Card"
)

#page style with a slightly scary hack
st.markdown("""
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
    </style>
""",
unsafe_allow_html=True)


### STREAMLIT APP LAYOUT###
#side bar setup
st.sidebar.title("Model Card")

#quant containers
quant_div = st.container()
quant_lcol, quant_rcol = st.columns([6,6])

#data contrainers
exp_div = st.container()


### STREAMLIT APP CONTENT ###
#load data
model_card = load_model_card()

#side panel
st.sidebar.header(f"{model_card['model-details']['name']}")
st.sidebar.write(f"{model_card['model-details']['short'][0]}")

with st.sidebar.expander("more details"):
    st.markdown(f"*{model_card['model-details']['short'][0]}")
    st.markdown(f"*{model_card['model-details']['short'][1]}")


st.header("Ethical Considerations")

#quant containers
with quant_div:
    with quant_lcol:
        st.write("Left Quant Column")
    with quant_rcol:
        st.write("Right Quant Column")

#adding own data
with exp_div:
    with st.expander("Add your Own Data"):
        st.write("Add your own data")






