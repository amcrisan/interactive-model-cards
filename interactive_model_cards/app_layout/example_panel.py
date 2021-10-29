import streamlit as st
import robustnessgym as rg
import pandas as pd
from math import floor
from random import sample


from interactive_model_cards import utils as ut


def format_data (user_text,model):
    ''' Formatting and preparing the user's input data'''

     # adding user data to the data panel
    dp = rg.DataPanel({"sentence": [user_text], "label": [1]})

    # run prediction
    dp, pred = ut.update_pred(dp,model)

    #summarizing the prediction

    idx_max = pred['Probability'].argmax()
    pred_sum = pred['Label'][idx_max]
    pred_bin = int(1) if pred['Label'][idx_max] == "Positive Sentiment" else int(0)
    pred_num = floor(pred['Probability'][idx_max] * 10 ** 3) / 10 ** 3
    pred_conf = ut.conf_level(pred['Probability'][idx_max])
    
    new_example = {
                    'sentence': user_text,
                    'model label':pred_sum,
                    'model label binary': pred_bin, 
                    'probability':pred_num,
                    'confidence' : pred_conf,
                    'user label': None,
                    'user label binary': None,
                }

    return(new_example)


def examples(col) :
    ''' UI for displaying the custom sentences'''

    #writing the metrics out to a column
    with col:
        st.markdown("** Custom Example Sentences **")
        
        if not st.session_state['user_data'].empty:
            #remove the user data slice

            #update the user data dev bench
            table = st.session_state['user_data'][['sentence','model label binary','user label binary',]]
            table.columns = ['sentence','pred','label']

            dp = rg.DataPanel({
                'sentence':table['sentence'].tolist(),
                'label':table['label'].tolist(),
                'pred': table['pred'].tolist()
            })

            dp._identifier = "Your Sentences"

            
            #updated the dev bench
            rg_bench  = ut.new_bench()
            rg_bench.add_slices(dp)

            #visualize the overall performance
            st.markdown("*Model Performance*")
            chart = ut.visualize_metrics(rg_bench.metrics['model'])

            st.altair_chart(chart, use_container_width=True)

            #add to overall model performance
            if st.button("Add Performance to Quant Panel"):
                st.session_state['quant_ex']['User Custom Sentence']= rg_bench.metrics['model']
            

            #visualize examples
            st.markdown("*Examples*")
            st.table(st.session_state['user_data'][['sentence','model label','user label','probability']])
        else:
            st.write("No examples added yet")



def example_sentence(col,sentence_examples,model):
    ''' UI for creating a custom sentences'''
    with col:

        # **** Entering Text *** 
        placeholder = st.empty()
        user_text= placeholder.text_input(
            "Write your own example sentences, or click 'Generate Examples Button'", 
            st.session_state['example_sent']
        )
        
        gen_button = st.button("Generate Examples", key="user_text")

        if gen_button :
            st.session_state['example_sent'] = sample(set(sentence_examples['sentences']),1)[0]
            
            user_text = placeholder.text_input(
            "Write your own example sentences, or click 'Generate Examples Button'", 
            st.session_state['example_sent']
        )


        if user_text !="":

            new_example = format_data(user_text,model)

            # **** Prediction Sumamary ***
            with st.form(key="my_form"):
                st.markdown("**Model Prediction Summary**")
                st.markdown(f"*The sentiment model predicts that this sentence has an overall `{new_example['model label']}` with an `{new_example['confidence']}` (p={new_example['probability']})*")
                
                #prediction agreement solicitation
                st.markdown("**Do you agree with the prediction?**")
                agreement = st.radio("Indicate your agreement below", ["Agree", "Disagree"])

                #getting the user label
                user_lab = new_example['model label']
                user_lab_bin = int(1) if new_example['model label'] == "Positive Sentiment" else int(0) 
                
                if agreement != "Agree":
                    user_lab = "Negative Sentiment" if new_example['model label'] =="Positive Sentiment" else "Positive Sentiment"
                    user_lab_bin = int(0) if user_lab_bin == 1 else int(1)


                #update robustness gym with user_example prediction
                if st.form_submit_button('Add to Example Sentences'):
                    #updating the user data frame
                    if user_text !="":
                        new_example['user label'] = user_lab
                        new_example['user label binary'] = user_lab_bin

                        #data frame to append to session info
                        new_example = pd.DataFrame(new_example,index=[0])

                        #update the session
                        st.session_state['user_data']  = st.session_state['user_data'].append(new_example, ignore_index=True)


def example_panel(sentence_examples,model):
    ''' Layout for the custom example panel'''

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
        exp_lcol, exp_rcol = st.columns([4, 8])

        # Layouts for lcol
        if data_src == "From Training Data":
            exp_lcol.write("You training data")

        elif data_src == "From Your Data":
            exp_lcol.write("Loading your own data")
        else:
            #adding a column for user text input
            example_sentence(exp_lcol,sentence_examples,model)
            examples(exp_rcol)
            
                                
                        
                    






