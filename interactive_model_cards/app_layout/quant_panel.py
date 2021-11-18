# streamlit
import streamlit as st
from streamlit_vega_lite import altair_component
import base64


# data
import pandas as pd

# utils
from numpy import round
from interactive_model_cards import utils as ut


def perf_interact(type="model perf",min_size=0):
    """ Instructions for interacting with the view"""

    if type == "model perf":
        st.markdown(
            f"""
            <span>
                <img src="data:image/png;base64,{base64.b64encode(open("./assets/img/warning-black.png", "rb").read()).decode()}"> All subpopulations with <strong>fewer than {min_size}</strong> sentences are reporting potentially unreliable results. These are <strong style="color:red">identified with a red border</strong> around the bar.
            </span>
            """,
            unsafe_allow_html=True
        )
        st.markdown("") #just to space them out
        st.markdown(
            f"""
            <span>
                <img src="data:image/png;base64,{base64.b64encode(open("./assets/img/click.png", "rb").read()).decode()}"> Click on the bars to see example sentences.
            </span>
            """,
            unsafe_allow_html=True
        )

        st.markdown("") #just to space them out
    else:
        st.write("This visualization shows a representation of the data according to how similar two sentences are *relative to the data the model was trained on*.")

        st.markdown(
            f"""
            <span>
                <img src="data:image/png;base64,{base64.b64encode(open("./assets/img/click.png", "rb").read()).decode()}"> <strong>Here are ways to interact with this view</strong>:
            </span>
            """,
            unsafe_allow_html=True
        )

        st.write("* You can `zoom in and out` of the visualization")
        st.write("* You can `hover` over a data point to see the sentence and sentiment")
        st.write("*  You can `click on the legend` to emphasize subpopulations in the data according to positive of negative sentiment.")

        


def quant_panel(sst_db, embedding, col,data_view):
    """ Quantitative Panel Layout"""

    all_metrics = {}
    with col:
        if data_view == "Model Performance Metrics":
            st.warning("**Model Performance Metrics**")

            st.markdown("* Evaluation metrics include [accuracy](https://simple.wikipedia.org/wiki/Accuracy_and_precision), [precision](https://en.wikipedia.org/wiki/Precision_and_recall), and [recall](https://en.wikipedia.org/wiki/Precision_and_recall).")
            st.markdown(" * Performance is shown for the training and testing set, as well as special groups within this dataset that have been automatically associated with US protected groups")
        

            min_size = st.number_input("Flag (with a red border) subpopulations with fewer than the follow sentences:", value=100, min_value=30, max_value=10000)
            
            perf_interact(type="model perf",min_size=min_size)

            #st.write(f'* All subsamples with `fewer than {min_size} sentences` are reporting potentially unreliable results and are <span style="color:red; fontface:bold">flagged with red border</span>. Take extra care when interpretting this data.', unsafe_allow_html=True)
            #st.markdown("* Click on the bars to see examples of sentences")

            for key in st.session_state["quant_ex"]:
                tmp = st.session_state["quant_ex"][key]

                if tmp is not None:
                    for iKey in tmp.keys():
                        all_metrics[iKey] = {}
                        all_metrics[iKey]["metrics"] = tmp[iKey]
                        all_metrics[iKey]["source"] = key

                        if key == "Overall Performance":
                            #get the size of the dataset
                            idx = ut.get_sliceid(list(sst_db.slices)).index(iKey)
                            slice_data = list(sst_db.slices)[idx]

                            # write slice data to UI
                            df = ut.slice_to_df(slice_data)
                            all_metrics[iKey]["size"] = df.shape[0]

                            # due to the way slices are added
                            # this hack is required
                            if "RGDataset" in iKey:
                                all_metrics[iKey]["source"] = "Custom Slice"
                            elif "protected" in iKey:
                                all_metrics[iKey]["source"] = "US Protected Class"
                        else:
                            all_metrics[iKey]["size"] = st.session_state["user_data"].shape[0]

            # st.write(all_metrics)
            chart = ut.visualize_metrics(all_metrics, max_width=100, linked_vis=True,min_size=min_size)
            event_dict = altair_component(altair_chart=chart)

            # st.altair_chart(chart)

            # if something was clicked on, find out what it was
            if "name" in event_dict.keys():
                # identify what it was selected on
                st.session_state["selected_slice"] = {
                    "name": event_dict["name"][0],
                    "source": event_dict["source"][0],
                }

            if st.session_state["selected_slice"] is not None:
                get_selected = st.session_state["selected_slice"]["name"]

                #subsampling data from training data
                if st.session_state["selected_slice"]["source"] in [
                    "Overall Performance",
                    "Custom Slice",
                    "US Protected Class"
                ]:
                    selected = st.session_state["selected_slice"]["name"]
                    # get selected slice data
                    #st.write(ut.get_sliceid(list(sst_db.slices)))
                    idx = ut.get_sliceid(list(sst_db.slices)).index(selected)
                    slice_data = list(sst_db.slices)[idx]

                    # write slice data to UI
                    df = ut.slice_to_df(slice_data)
                

                #subsetting the data
                    st.warning("**Data Details**")
                    with st.expander("Customize Data Sample"):
                        with st.form("Sample Form"):
                            st.number_input(
                                "Number of Samples",
                                value=min(df.shape[0],10),
                                min_value=1,
                                max_value=df.shape[0],
                                key="sampleNum",
                            )
                            st.selectbox(
                                "Sample Type",
                                [
                                    "Random Sample",
                                    "Highest Probabilities",
                                    "Lowest Probabilities",
                                    "Mid Probabilities",
                                ],
                                index=0,
                                key="sampleType",
                            )
                            st.form_submit_button("Generate Sample")
                    
                    #drawing the sampled data
                    
                    #summarize slice information
                    displayName = str(selected).split("->")
                    
                    if len(displayName) > 1:
                        displayName = displayName[1].split("@")[0].strip()
                    else:
                        displayName= displayName[0]

                    st.markdown(
                        f"* The slice `{displayName}` has a total size of `{df.shape[0]} sentences`"
                    )
                    #summarize data sample size and sampling method
                    st.markdown(
                        f"* Shown is a subsample of all the data to `{st.session_state['sampleNum']}` sampled by `{st.session_state['sampleType']}`"
                    )

                    # add terms in user has selectd a custom slice
                    if st.session_state["selected_slice"]["source"]=="Custom Slice":
                        terms_str = ', '.join(st.session_state["slice_terms"][selected])
                        st.markdown(f"* This slice contains sentences containing one or more of following has the following terms:`{terms_str}`")

                    elif st.session_state["selected_slice"]["source"]=="US Protected Class":
                        terms = st.session_state["protected_class"][displayName]
                        terms_str = ", ".join(terms)
                        st.markdown(f"* Sentences pertaining this US Protected Classes contain the following-terms: `{terms_str}`")
                        st.markdown(
                            f"""
                            <span>
                                <img src="data:image/png;base64,{base64.b64encode(open("./assets/img/warning-black.png", "rb").read()).decode()}"> Detecting US Protected classess by key word search is not perfect. Some sentences below may not be pertintent to a protected class, for example the word 'black' can refer individuals but not always.
                            </span>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    st.table(
                        ut.subsample_df(
                            df,
                            st.session_state["sampleNum"],
                            st.session_state["sampleType"],
                        )
                    )

                elif st.session_state["selected_slice"]["source"] in ["User Custom Sentence"]:
                    #st.markdown(f"These are {st.session_state["user_data"]} custom sentences you have defined")
                    st.markdown("**Data Details**")
                    df = st.session_state["user_data"]
                    st.markdown(f"These are your `{df.shape[0]}` custom sentences")
                    st.write(df)
        else:
            st.warning("**Subpopulation Comparison**")
            perf_interact(type="comparison")

            with st.expander("how to read this chart:"):
                st.markdown("* each **point** is a single sentence")
                st.markdown("* the **position** of each dot is determined mathematically based upon an analysis of the words in a sentence. The **closer** two points on the visualization the **more similar** the sentences are. The **further apart ** two points on the visualization the **more different** the sentences are")
                st.markdown(" * the **shape** of each point reflects whether it a positive (diamond) or negative sentiment (circle)")
                st.markdown("* the **color** of each point is the ")

            #down sample embedding for altair limitations
            tmp = embedding
            tmp = ut.down_samp(embedding)
            st.altair_chart(ut.data_comparison(tmp))


__all__ = ["quant_panel"]
