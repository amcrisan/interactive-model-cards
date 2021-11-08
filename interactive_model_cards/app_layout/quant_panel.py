# streamlit
import streamlit as st
from streamlit_vega_lite import altair_component

# data
import pandas as pd

# utils
from numpy import round
from interactive_model_cards import utils as ut


def quant_panel(sst_db, embedding, col):
    """ Quantitative Panel Layout"""

    all_metrics = {}
    with col:
        data_view = st.selectbox("Show:",["Model Performance Metrics","Data Subpopulation Comparison Visualization"])
        st.markdown("-----")

        if data_view == "Model Performance Metrics":
            st.warning("**Model Performance Metrics**")
            st.write("The performance of the model is broken down into different subpopulations of the training and test data. Performance is assessed via, accuracy, precision, and recall metric. The size of the population reflects the reliability of how these preformance metrics are calculated. The minimal sample size is an adjustable value to help identify small subpopulations in the data. ")
            min_size = st.number_input("Minimal Sample Size:", value=100, min_value=30, max_value=10000)
            st.write(f'* All subsamples with `fewer than {min_size} sentences` are reporting potentially unreliable results and are <span style="color:red; fontface:bold">flagged with red border</span>. Take extra care when interpretting this data.', unsafe_allow_html=True)
            st.markdown("* Click on the bars to see examples of sentences")

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


            #----- V OUTSIDE OF COLUMN  v -----
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
                

                #----- V OUTSIDE OF COLUMN  v -----
            #with col:
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
                    st.markdown(
                        f"* The slice `{selected}` has a total size of `{df.shape[0]} sentences`"
                    )
                    # add terms in user has selectd a custom slice
                    if st.session_state["selected_slice"]["source"]=="Custom Slice":
                        terms_str = ', '.join(st.session_state["slice_terms"][selected])
                        st.markdown(f"* This slice contains sentences containing one or more of following has the following terms:`{terms_str}`")

                    #summarize data sample size and sampling method
                    st.markdown(
                        f"* Shown is a subsample of all the data to `{st.session_state['sampleNum']}` sampled by `{st.session_state['sampleType']}`"
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
            
            st.write("This visualization shows a representation of the data according to how similar two sentences are relative to the data the model was trained on. The **closer** two points on the visualization the **more similar** the sentences are.")

            st.write("You can explore this visualization in two ways:")
            st.write(" * You can `zoom in and out` of the visualization")
            st.write(" * You can `click on the legend` to emphasize subpopulations in the data according to positive of negative sentiment.")
            st.write(" * You can `hover` over a data point to see the sentence and sentiment")

            #down sample embedding for altair limitations
            tmp = embedding
            tmp = ut.down_samp(embedding)
            st.altair_chart(ut.data_comparison(tmp))

__all__ = ["quant_panel"]
