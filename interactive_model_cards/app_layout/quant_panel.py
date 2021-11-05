#streamlit
import streamlit as st
from streamlit_vega_lite import altair_component

#data
import pandas as pd

#utils
from numpy import round
from interactive_model_cards import utils as ut


def quant_panel(sst_db,col):
    """ Quantiative Panel Layout"""

    #quant_lcol, quant_rcol = st.columns([8, 4])

    all_metrics = {}
    with col:
        st.write(
        """<h1 style="font-size:20px"> Quantitative Examples</h1>""",
        unsafe_allow_html=True)
        for key in st.session_state["quant_ex"]:
            tmp = st.session_state["quant_ex"][key]

            if tmp is not None:
                for iKey in tmp.keys():
                    all_metrics[iKey] = {}
                    all_metrics[iKey]['metrics'] = tmp[iKey]
                    all_metrics[iKey]['source'] = key

                    if key == "Overall Performance":
                        #due to the way slices are added
                        #this hack is required
                        if "RGDataset" in iKey:
                            all_metrics[iKey]['source'] = "Custom Slice"
                        


        #st.write(all_metrics)
        chart = ut.visualize_metrics(all_metrics, max_width=100,linked_vis=True)
        event_dict = altair_component(altair_chart=chart)
       
        #st.altair_chart(chart)

    #if something was clicked on, find out what it was
    if 'name' in event_dict.keys():
        #identify what it was selected on
        st.session_state['selected_slice'] = {'name': event_dict['name'][0],
                                            'source': event_dict['source'][0]}
    
    if st.session_state['selected_slice'] is not None:
        get_selected = st.session_state['selected_slice']['name']

        if st.session_state['selected_slice']['source'] in ["Overall Performance","Custom Slice"]:
            selected = st.session_state['selected_slice']['name']
            
            #get selected slice data
            idx = ut.get_sliceid(list(sst_db.slices)).index(selected)
            slice_data = list(sst_db.slices)[idx]

            #write slice data to UI
            #quant_rcol.dataframe(ut.slice_to_df(slice_data))
            df = ut.slice_to_df(slice_data)
            
            with col:
                st.write("**Data Details**")
                with st.expander("Customize Data Sample"):
                    with st.form("Sample Form"):
                        st.number_input("Number of Samples", 
                            value=10, 
                            min_value=1, 
                            max_value=df.shape[0],
                            key = "sampleNum")
                        st.selectbox("Sample Type", 
                            ["Random Sample","Highest Probabilities","Lowest Probabilities","Mid Probabilities"], 
                            index=0,
                            key = "sampleType")
                        st.form_submit_button("Generate Sample")

                        
                
                st.markdown(f"* The slice `{selected}` has a total size of `{df.shape[0]} sentences`")
                st.markdown(f"* Shown is a subsample of all the data to `{st.session_state['sampleNum']}` sampled by `{st.session_state['sampleType']}`")

                st.table(ut.subsample_df(df,
                    st.session_state['sampleNum'],
                    st.session_state['sampleType'])
                )

                
            
        else:
            col.table(st.session_state['user_data'][['sentence','model label','user label']])
