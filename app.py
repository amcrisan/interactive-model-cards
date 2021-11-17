### LIBRARIES ###
# # Data
import numpy as np
import pandas as pd
import json
from math import floor

# Robustness Gym and Analysis
import robustnessgym as rg
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# App & Visualization
import streamlit as st
import altair as alt

# utils
from interactive_model_cards import utils as ut
from interactive_model_cards import app_layout as al
from random import sample
from PIL import Image

### LOADING DATA ###
# model card data
@st.experimental_memo
def load_model_card():
    with open("./assets/data/text_explainer/model_card.json") as f:
        mc_text = json.load(f)
    return mc_text


# pre-computed robusntess gym dev bench
# @st.experimental_singleton
@st.cache(allow_output_mutation=True)
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

#load pre-computed embedding
def load_embedding():
    embedding = pd.read_pickle("./assets/models/sst_vectors.pkl")
    return embedding

#load doc2vec model
@st.experimental_singleton
def load_doc2vec():
    doc2vec = Doc2Vec.load("./assets/models/sst_train.doc2vec")
    return(doc2vec)  
    

# @st.experimental_memo
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
    #protected_classes 
    protected_classes = json.load(open("./assets/data/protected_terms.json"))
    
    return devBench, model, protected_classes

@st.experimental_singleton
def load_title():
    img = Image.open("./assets/img/title.png")
    return(img)


if __name__ == "__main__":

    ### STREAMLIT APP CONGFIG ###
    st.set_page_config(layout="wide", page_title="Interactive Model Card")

    # import custom styling
    ut.init_style()

    ### LOAD DATA AND SESSION VARIABLES ###

    # ******* loading the mode and the data
    with st.spinner():
        sst_db, model,protected_classes = load_basic()
        embedding = load_embedding()
        doc2vec = load_doc2vec()

    # load example sentences
    sentence_examples = load_examples()

    # ******* session state variables
    if "user_data" not in st.session_state:
        st.session_state["user_data"] = pd.DataFrame()
    if "example_sent" not in st.session_state:
        st.session_state["example_sent"] = "I like you. I love you"
    if "quant_ex" not in st.session_state:
        st.session_state["quant_ex"] = {"Overall Performance": sst_db.metrics["model"]}
    if "selected_slice" not in st.session_state:
        st.session_state["selected_slice"] = None
    if "slice_terms" not in st.session_state:
        st.session_state["slice_terms"] = {}
    if "embedding" not in st.session_state:
        st.session_state["embedding"] = embedding
    if "protected_class" not in st.session_state:
        st.session_state["protected_class"] = protected_classes


    ### STREAMLIT APP LAYOUT###

    # ******* MODEL CARD PANEL *******
    #st.sidebar.title("Interactive Model Card")
    img = load_title()
    st.sidebar.image(img,width=400)
    st.sidebar.warning("Data is not permanently collected or stored from your interactions, but is temporarily cached during usage.")

    # load model card data
    model_card = load_model_card()
    al.model_card_panel(model_card)

    lcol, rcol = st.columns([4, 8])

    # ******* USER EXAMPLE DATA PANEL *******
    st.markdown("---")
    with lcol:

        # Choose waht to show for the qunatiative analysis.
        st.write("""<h1 style="font-size:20px;padding-top:0px;"> Quantitative Analysis</h1>""",
                unsafe_allow_html=True)
        
        st.markdown("View the model's performance or visually explore the model's training and testing dataset")

        data_view = st.selectbox("Show:",
            ["Model Performance Metrics","Data Subpopulation Comparison Visualization"])
            
        st.markdown("Any groups you define via the *analysis actions* will be automatically added to the view")
        st.markdown("---")

        
        # Additional Analysis Actions
        st.write(
            """<h1 style="font-size:18px;padding-top:5px;"> Analysis Actions</h1>""",
            unsafe_allow_html=True,
        )
        al.example_panel(sentence_examples, model, sst_db,doc2vec)

    # ****** GUIDANCE PANEL *****
        with st.expander("Guidance"):
            st.markdown(
                "Need help understanding what you're seeing in this model card?"
            )

            st.markdown(
                " * **[Understanding Metrics](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)**: A cheatsheet of model metrics"
            )
            st.markdown(
                " * **[Understanding Sentiment Models](https://www.semanticscholar.org/topic/Sentiment-analysis/6011)**: An overview of sentiment analysis"
            )
            st.markdown(
                "* **[Next Steps](https://docs.google.com/document/d/1r9J1NQ7eTibpXkCpcucDEPhASGbOQAMhRTBvosGu4Pk/edit?usp=sharin)**: Suggestions for follow-on actions"
            )
            st.markdown("Feel free to submit feedback via our [online form](https://sfdc.co/imc_feedback)")
    
    # ******* QUANTITATIVE DATA PANEL *******
    al.quant_panel(sst_db, st.session_state["embedding"], rcol,data_view)
