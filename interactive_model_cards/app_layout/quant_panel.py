# streamlit
import streamlit as st
from streamlit_vega_lite import altair_component

# data
import pandas as pd

# utils
from numpy import round
from interactive_model_cards import utils as ut


def quant_panel(sst_db, col):
    """ Quantiative Panel Layout"""

    all_metrics = {}
    with col:
        min_size = st.number_input("Minimal Sample Size:", value=1000, min_value=30, max_value=10000)
        st.markdown(f"*All subsamples with `fewer than {min_size} sentences` are reporting potentially unreliable results and are `flagged with a red border`. Take extra care when interpretting this data.*")

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
            "Custom Slice"
        ]:
            selected = st.session_state["selected_slice"]["name"]
            # get selected slice data
            st.write(ut.get_sliceid(list(sst_db.slices)))
            idx = ut.get_sliceid(list(sst_db.slices)).index(selected)
            slice_data = list(sst_db.slices)[idx]

            # write slice data to UI
            df = ut.slice_to_df(slice_data)
            with col:
                #subsetting the data
                st.markdown("**Data Details**")
                with st.expander("Customize Data Sample"):
                    with st.form("Sample Form"):
                        st.number_input(
                            "Number of Samples",
                            value=10,
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


                #drawing the sampled data
                st.table(
                    ut.subsample_df(
                        df,
                        st.session_state["sampleNum"],
                        st.session_state["sampleType"],
                    )
                )

        elif st.session_state["selected_slice"]["source"] in ["User Custom Sentence"]:
            with col:
                #st.markdown(f"These are {st.session_state["user_data"]} custom sentences you have defined")
                st.markdown("**Data Details**")
                df = st.session_state["user_data"]
                st.markdown(f"These are your `{df.shape[0]}` custom sentences")
                st.write(df)

__all__ = ["quant_panel"]
