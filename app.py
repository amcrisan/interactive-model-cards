### LIBRARIES ###

# App
import streamlit as st

# Data
import numpy as np
import pandas as pd
import json

#Visualization


### LOAD DATA ###
#model card
with open('./data/text_explainer/model_card.json') as f:
  model_card = json.load(f)

### STREAMLIT APP ###
st.set_page_config(layout="wide")
left_col, right_col = st.columns([4,8])


with left_col:
    st.header("Model Information")
    st.text("Click on (+) for additional information")
    with st.expander(model_card['model-details']['name']):
        st.write(model_card['model-details']['short'])

    with st.expander(model_card['intended-use']['name']):
        st.write(model_card['intended-use']['short'])

    with st.expander(model_card['ethical-considerations']['name']):
        st.write(model_card['ethical-considerations']['short'])

   

with right_col:
    st.header("Quantitative Analysis")
    st.text('I am the right column') 


#sidebar
st.sidebar.header("Add Your Own Data")
st.sidebar.selectbox('Which number do you like best?',[1,2,3,4,5])