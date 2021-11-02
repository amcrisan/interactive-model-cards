import streamlit as st
import robustnessgym as rg
import pandas as pd
from math import floor
from random import sample


from interactive_model_cards import utils as ut


def format_data(user_text, model):
    """ Helper Function : Formatting and preparing the user's input data"""

    # adding user data to the data panel
    dp = rg.DataPanel({"sentence": [user_text], "label": [1]})

    # run prediction
    dp, pred = ut.update_pred(dp, model)

    # summarizing the prediction

    idx_max = pred["Probability"].argmax()
    pred_sum = pred["Label"][idx_max]
    pred_bin = int(1) if pred["Label"][idx_max] == "Positive Sentiment" else int(0)
    pred_num = floor(pred["Probability"][idx_max] * 10 ** 3) / 10 ** 3
    pred_conf = ut.conf_level(pred["Probability"][idx_max])

    new_example = {
        "sentence": user_text,
        "model label": pred_sum,
        "model label binary": pred_bin,
        "probability": pred_num,
        "confidence": pred_conf,
        "user label": None,
        "user label binary": None,
    }

    return new_example


def slice_misc(table):
    ''' Helper Function: format new slice'''
    table = st.session_state["user_data"][
                ["sentence", "model label binary", "user label binary"]
            ]
    table.columns = ["sentence", "pred", "label"]

    dp = rg.DataPanel(
        {
            "sentence": table["sentence"].tolist(),
            "label": table["label"].tolist(),
            "pred": table["pred"].tolist(),
        }
    )

    #give the sentence a name
    dp._identifier = "Your Sentences"

    # updated the dev bench
    rg_bench = ut.new_bench()
    rg_bench.add_slices(dp)

    return(rg_bench)



# ***** ADDING CUSTOM SENTENCES *******
def examples(col):
    """ UI for displaying the custom sentences"""

    # writing the metrics out to a column
    with col:
        st.markdown("** Custom Example Sentences **")

        if not st.session_state["user_data"].empty:
            # remove the user data slice

            # visualize the overall performance
            st.markdown("*Model Performance*")
            key = "Your Sentences"
            all_metrics = {key:{}}
            all_metrics[key]["metrics"] = st.session_state["quant_ex"]["User Custom Sentence"][key]
            all_metrics[key]["source"] = key

            #chart = ut.visualize_metrics(st.session_state["quant_ex"]["User Custom Sentence"])
            chart = ut.visualize_metrics(all_metrics,col_val= "#ff7f0e")
            st.altair_chart(chart)

            # add to overall model performance
            # visualize examples
            st.markdown("*Examples*")
            st.dataframe(
                st.session_state["user_data"][
                    ["sentence", "model label", "user label", "probability"]
                ]
            )
        else:
            st.write("No examples added yet")


def example_sentence(col, sentence_examples, model):
    """ UI for creating a custom sentences"""
    with col:

        # **** Entering Text ***
        placeholder = st.empty()
        user_text = placeholder.text_input(
            "Write your own example sentences, or click 'Generate Examples Button'",
            st.session_state["example_sent"]
        )

        gen_button = st.button("Generate Examples", key="user_text")

        if gen_button:
            st.session_state["example_sent"] = sample(
                set(sentence_examples["sentences"]), 1
            )[0]

            user_text = placeholder.text_input(
                "Write your own example sentences, or click 'Generate Examples Button'",
                st.session_state["example_sent"],
            )

        if user_text != "":

            new_example = format_data(user_text, model)

            # **** Prediction Sumamary ***
            with st.form(key="my_form"):
                st.markdown("**Model Prediction Summary**")
                st.markdown(
                    f"*The sentiment model predicts that this sentence has an overall `{new_example['model label']}` with an `{new_example['confidence']}` (p={new_example['probability']})*"
                )

                # prediction agreement solicitation
                st.markdown("**Do you agree with the prediction?**")
                agreement = st.radio(
                    "Indicate your agreement below", ["Agree", "Disagree"]
                )

                # getting the user label
                user_lab = new_example["model label"]
                user_lab_bin = (
                    int(1)
                    if new_example["model label"] == "Positive Sentiment"
                    else int(0)
                )

                if agreement != "Agree":
                    user_lab = (
                        "Negative Sentiment"
                        if new_example["model label"] == "Positive Sentiment"
                        else "Positive Sentiment"
                    )
                    user_lab_bin = int(0) if user_lab_bin == 1 else int(1)

                # update robustness gym with user_example prediction
                if st.form_submit_button("Add to Example Sentences"):
                    # updating the user data frame
                    if user_text != "":
                        new_example["user label"] = user_lab
                        new_example["user label binary"] = user_lab_bin

                        # data frame to append to session info
                        new_example = pd.DataFrame(new_example, index=[0])

                        # update the session
                        st.session_state["user_data"] = st.session_state[
                            "user_data"
                        ].append(new_example, ignore_index=True)

                        # update the user data dev bench
                        user_bench = slice_misc(st.session_state["user_data"])

                        #add bench
                        st.session_state["quant_ex"]["User Custom Sentence"] = user_bench.metrics["model"]



# ***** DEFINTING CUSTOM SUBGROUPS *******

def subpopulation_slice(col,sst_db):
    with col:
        with st.form(key="subpop_form"):
            st.markdown("Define you subpopulation")
            user_terms = st.text_input("Enter a set of comma separated words","comedy, hilarious, clown")
            slice_choice = st.selectbox("Choose Data Source", ["Training Data","Test Data"])
            slice_name = st.text_input("Give your subpopulation a name","supbob_1")
            if st.form_submit_button("Create Subpopulation"):
                #build a new slice
                #
                user_terms = [x.strip() for x in user_terms.split(',')]
                slice_builder = rg.HasAnyPhrase([user_terms], identifiers=[slice_name])
                
                #on test data
                sst_db(slice_builder, list(sst_db.slices)[0],['sentence'])
                
                #return terms
                return(user_terms)

def slice_vis(col,terms,sst_db):
    with col:
        st.write(terms)
        #TO DO - FORMATTING AND ADD METRICS
        if len(list(sst_db.slices))>2:
            #write out the dataset for this subset
            st.write(list(sst_db.slices)[2][['sentence','label']])



# ***** EXAMPLE PANEL UI *******    

def example_panel(sentence_examples,model,sst_db):
    """ Layout for the custom example panel"""

    # Data Expander
    with st.expander("Add your own examples to test the model on!", expanded=True):
        
        # layouts for the expander
        exp_lcol, exp_rcol = st.columns([4, 8])

        data_src = exp_lcol.selectbox(
            "Select Example Source",
            ["Help","Text Example", "From Model Data", "From Your Data"],
        )
        # Title
        title = "Add your own sentences as Examples"
        if data_src == "From Model Data":
            title = "Create new subset's from the model's data"
        elif data_src == "From Your Data":
            title = "Load your own Data Set"
        elif data_src == "Text Example":
            title = "Add a Text Example"
        elif data_src == "Help":
            title = "Help"

        exp_lcol.markdown(f"** {title} **")

        # Layouts for lcol
        if data_src == "Help":
            with exp_lcol:
                st.markdown("Here's an overview of the ways you can add customized the performance results. Using the drop down menu above, you can choose from one of three options")
                st.markdown("1. **Text Example** : Add your own sentences as examples")
                st.markdown("2. **From Model Data** : Create a new subset from the model's training or testing data")
                st.markdown("3. **From your Data** : Upload your own (small) dataset from a csv file")

        elif data_src == "From Model Data":
            slice_terms = subpopulation_slice(exp_lcol,sst_db)
            slice_vis(exp_rcol,slice_terms,sst_db)

        elif data_src == "From Your Data":
            exp_lcol.write("Loading your own data")
        else:
            # adding a column for user text input
            example_sentence(exp_lcol, sentence_examples, model)
            examples(exp_rcol)
