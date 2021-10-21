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

    div.row-widget.stRadio > div{flex-direction:row;}
    </style>
""",
unsafe_allow_html=True)


### STREAMLIT APP LAYOUT###
#******* side bar setup
st.sidebar.title("Interactive Model Card")

#load model card data
model_card = load_model_card()

#side panel
st.sidebar.markdown(f"**{model_card['model-details']['name']}**")
st.sidebar.write(f"{model_card['model-details']['short'][0]}")

with st.sidebar.expander("more details"):
    st.markdown(f"*{model_card['model-details']['short'][0]}")
    st.markdown(f"*{model_card['model-details']['short'][1]}")



#******* quantaitive analysis
st.markdown('''**Quantaitive Analysis**''')
quant_lcol, quant_rcol = st.columns([6,6])

with quant_lcol:
    st.write("Left Column")

with quant_rcol:
    st.write("Right Column")


#******* Example Layout 
with st.expander("Add your own examples"):
    data_src = st.radio('Select Example Source', 
    ['Text Example','From Training Data','From Your Data'])
    st.markdown("""---""")

    exp_lcol, exp_mid, exp_rcol = st.columns([5,5,2])

    if data_src == "From Training Data":
        exp_lcol.write("You training data")
        exp_mid.write("You training data 2")
        exp_rcol.write("You training data 3" )

    elif data_src=="From Your Data":
        exp_lcol.write("Loading your own data")
        exp_mid.write("Loading your own data 2")
        exp_rcol.write("Loading your own data 3" )
    else:
        exp_lcol.text_input("Add your own example text",'I like you. I love you')
        exp_mid.write("Example Text 2")

#adding own data
#with exp_div:
#    with st.expander("Add your Own Examples"):
#        data_src = st.radio('Select Example Source', 
#                       ['Text Example','From Training Data','From Your Data'])

#        if data_src == "From Training Data":
#            st.write("You training data")
#        elif data_src=="From Your Data":
#            st.write("Loading your own data")
#        else:
#            st.write("Example Text")


#quant containers
#with quant_div:
#    with quant_lcol:
#        st.write("Left Quant Column")
#    with quant_rcol:
#        st.write("Right Quant Column")





