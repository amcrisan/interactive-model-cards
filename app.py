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
import utils as ut
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



@st.experimental_memo
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


### STREAMLIT APP CONGFIG ###
st.set_page_config(layout="wide", page_title="Interactive Model Card")

#import custom styling
ut.init_style()



### STREAMLIT APP LAYOUT###
# ******* side bar setup
st.sidebar.title("Interactive Model Card")

# load model card data
model_card = load_model_card()


# model card side panel
for key in model_card.keys():
    item = model_card[key]
    st.sidebar.markdown(f"**{model_card[key]['name']}**")
    st.sidebar.write(f"{model_card[key]['short'][0]}")

    with st.sidebar.expander("more details"):
        for detail in model_card[key]['short']:
            st.markdown(f"* {detail}")

#loading the mode and the data
with st.spinner():
    sst_db, model = load_basic()

#load example sentences
sentence_examples = load_examples()


# ******* QUANTITATIVE DATA PANEL ******* 
st.write("""<h1 style="font-size:20px"> Quantitative Examples</h1>""", unsafe_allow_html=True)

quant_lcol, quant_rcol = st.columns([6, 6])

with quant_lcol:
    #st.write(sst_db.metrics)
    st.write("Left Column")
with quant_rcol:
    st.write("Right Column")


# ******* USER EXAMPLE DATA PANEL ******* 
st.markdown("---")
st.write("""<h1 style="font-size:20px;padding-top:0px;"> Additional Examples</h1>""", unsafe_allow_html=True)

#EXAMPLE SESSIONS VARIABLES
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = pd.DataFrame()
if 'example_sent' not in st.session_state:
    st.session_state['example_sent'] = sample(set(sentence_examples['sentences']),1)
    

# Data Expander
with st.expander("Add your own examples to test the model on!", expanded=True):
    data_src = st.selectbox(
        "Select Example Source",
        ["Text Example", "From Training Data", "From Your Data"],
    )
    #Title
    title = "Add your own sentences as Examples"
    if data_src == "From Training Data":
        title = "Create New Susbsets from the Training Data"
    elif data_src == "From Your Data":
        title = "Load your own Data Set"

    st.markdown(f"** {title} **")


    #layouts for the expander
    exp_lcol, exp_rcol = st.columns([5, 7])

    # Layouts for lcol
    if data_src == "From Training Data":
        exp_lcol.write("You training data")

    elif data_src == "From Your Data":
        exp_lcol.write("Loading your own data")
    else:
        #adding a column for user text input
       
        with exp_lcol:
            user_text = st.text_input(
                "Write your own example sentences, or click 'Generate Examples Button'", f"{st.session_state['example_sent'][0]}",
                key="user_text"
            )
            #update the example
            if st.button("Generate New Example"):
                st.session_state['example_sent']= sample(set(sentence_examples['sentences']),1)
            if user_text !="":
                # adding user data to the data panel
                dp = rg.DataPanel({"sentence": [user_text], "label": [1]})
                dp._identifier = f"User Input - {user_text}"

                # run prediction
                dp, pred = ut.update_pred(dp,model)
            
                #summarizing the prediction
               
                idx_max = pred['Probability'].argmax()
                pred_sum = pred['Label'][idx_max]
                pred_num = floor(pred['Probability'][idx_max] * 10 ** 3) / 10 ** 3
                pred_conf = ut.conf_level(pred['Probability'][idx_max])
                
                with st.form(key="my_form"):
                    st.markdown("**Model Prediction Summary**")
                    st.markdown(f"*The sentiment model predicts that this sentence has an overall `{pred_sum}` with an `{pred_conf}` (p={pred_num})*")
                    
                    #prediction agreement solicitation
                    st.markdown("**Do you agree with the prediction?**")
                    agreement = st.radio( "Indicate your agreement below", [f"Agree", "Disagree"])
                    st.write(f"You `{agreement}` with the models prediction of `{pred_sum}`")

                    #getting the user label
                    user_lab = pred_sum
                    user_lab_bin = round(pred_num)
                    if agreement != "Agree":
                        user_lab = "Negative Sentiment" if pred_sum =="Positive Sentiment" else "Positive Sentiment"
                        user_lab_bin = int(0) if user_lab_bin == 1 else int(1)


                    #update robustness gym with user_example prediction
                    if st.form_submit_button('Add to Example Sentences'):
                        #updating the user data frame
                        if user_text !="":
                            new_example = pd.DataFrame({
                                'sentence': user_text,
                                'model label':pred_sum,
                                'user label': user_lab,
                                'user label binary': user_lab_bin,
                                'probability':pred_num
                            },index=[0])

                            #update the session
                            st.session_state['user_data']  = st.session_state['user_data'].append(new_example, ignore_index=True)

                    
                
        #writing the metrics out to a column
        with exp_rcol:
            st.markdown("** Custom Example Sentences **")
            
            if not st.session_state['user_data'].empty:
                #remove the user data slice
                #ut.remove_slice(sst_db,"RGDataset")

                #add the latest user data slice
                table = st.session_state['user_data'][['sentence','probability','user label binary',]]
                table.columns = ['sentence','label','pred']

                dp = rg.DataPanel({
                'sentence':table['sentence'].tolist(),
                'label':table['pred'].tolist(),
                'pred': table['label'].round().tolist()})

                #st.write(dp)
                sst_db.add_slices(dp)
                st.write(sst_db.metrics)


            
            #vis_tab = st.session_state['user_data'][['sentence','model label','user label', 'probability']]
            #smol_lcol,smol_rcol = st.columns([3,3])
            st.table(st.session_state['user_data'])


                
