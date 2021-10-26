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


### LOADING DATA ###

# model card data
@st.experimental_memo
def load_model_card():
    with open("./assets/data/text_explainer/model_card.json") as f:
        mc_text = json.load(f)
    return mc_text


# pre-computed robusntess gym dev bench
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


#updating a prediction

def update_pred(dp,model):
    ''' Updating data panel with model prediction'''

    model.predict_batch(dp, ["sentence"])
    dp = dp.update(
            lambda x: model.predict_batch(x, ["sentence"]),
            batch_size=4,
            is_batched_fn=True,
            pbar=True,
    )

    labels = pd.Series(['Negative Sentiment','Positive Sentiment'])
    probs = pd.Series(dp.__dict__["_data"]["probs"][0])
    
    pred = pd.concat([labels, probs], axis=1)
    pred.columns = ['Label','Probability']


    return(dp, pred)

def conf_level(val):
        ''' Translates probability value into
        a plain english statement '''
        #https://www.dni.gov/files/documents/ICD/ICD%20203%20Analytic%20Standards.pdf
        conf = 'undefined'
        print(val)
        if val < 0.05:
            conf= 'Extremely Low Probability'
        elif val >=0.05 and val <0.20:
            conf = "Very Low Probability"
        elif val >=0.20 and val <0.45:
            conf = "Low Probability"
        elif val >=0.45 and val <0.55:
            conf = "Middling Probability"
        elif val >=0.55 and val <0.80:
            conf = "High  Probability"
        elif val >=0.80 and val <0.95:
            conf = "Very High Probability"
        elif val >=0.95:
            conf = "Extremely High Probability"
        
        return(conf)
    

def add_slice(bench,slice):
    ''' Add a slice to the dev bench'''
    return(bench.add_slices([slice]))

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

    /* Radio Button Direction*/
    div.row-widget.stRadio > div{flex-direction:row;}

    /* Expander Boz*/
    .streamlit-expander {
        border-width: 0px;
        border-bottom: 1px solid #A29C9B;
        border-radius: 0px;
    }

    .streamlit-expanderHeader {
        font-style: italic;
        font-weight :600;
        padding-top:0px;
        padding-left: 0px;
        color:#A29C9B

    /* Section Headers */
    .sectionHeader {
        font-size:10px;
    }
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
st.write("""<h1 style="font-size:20px"> Quantaitive Examples</h1>""", unsafe_allow_html=True)

quant_lcol, quant_rcol = st.columns([6, 6])

with quant_lcol:
    st.write(sst_db.metrics)
with quant_rcol:
    st.write("Right Column")


# ******* Example Layout
st.write("""<h1 style="font-size:20px;padding-top:0px;"> Additional Examples</h1>""", unsafe_allow_html=True)

with st.expander("Add your own examples to test the model on!", expanded=True):
    data_src = st.radio(
        "Select Example Source",
        ["Text Example   ", "From Training Data   ", "From Your Data   "],
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
        #adding a column for user text input
        with exp_lcol:
            user_text = st.text_input(
                "Write your own example sentences", "I like you. I love you"
            )

            # adding user data to the data panel
            dp = rg.DataPanel({"sentence": [user_text], "label": [1]})
            dp._identifier = f"User Input - {user_text}"

            # run prediction
            dp, pred = update_pred(dp,model)
        
            #summarizing the prediction
            st.markdown("**Model Prediction Summary**")
            idx_max = pred['Probability'].argmax()
            pred_sum = pred['Label'][idx_max]
            pred_num = floor(pred['Probability'][idx_max] * 10 ** 3) / 10 ** 3
            pred_conf = conf_level(pred['Probability'][idx_max])

            st.markdown(f"*The sentiment model predicts that this sentence has an overall `{pred_sum}` with an `{pred_conf}` (p={pred_num})*")
            
            #prediction agreement solicitation
            st.markdown("**Do you agree with the prediction?**")
            agreement = st.radio( "Indicate your agreement below", [f"Agree", "Disagree"])
            st.write(f"You `{agreement}` with the models prediction of `{pred_sum}`")

            #add example as slice
           # st.button('Add Example',on_click = add_slice(sst_db,dp))            
        
        #writing the metrics out to a column
        exp_mid.write(sst_db.metrics)
