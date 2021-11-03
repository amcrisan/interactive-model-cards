#streamlit
import streamlit as st
from streamlit_vega_lite import altair_component

#data
import pandas as pd

#utils
from numpy import round
from interactive_model_cards import utils as ut


def quant_panel(sst_db):
    """ Quantiative Panel Layout"""
    st.write(
        """<h1 style="font-size:20px"> Quantitative Examples</h1>""",
        unsafe_allow_html=True,
    )

    quant_lcol, quant_rcol = st.columns([6, 6])

    all_metrics = {}
    with quant_lcol:
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
        get_selected = event_dict['name'][0]
        
        if event_dict['source'][0] in ["Overall Performance","Custom Slice"]:
            selected = event_dict['name'][0]
            
            #get selected slice data
            idx = ut.get_sliceid(list(sst_db.slices)).index(selected)
            slice_data = list(sst_db.slices)[idx]

            #write slice data to UI
            quant_rcol.dataframe(ut.slice_to_df(slice_data))
            
        else:
            quant_rcol.table(st.session_state['user_data'][['sentence','model label','user label']])
