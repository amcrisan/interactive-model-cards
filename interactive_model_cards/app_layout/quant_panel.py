import streamlit as st
from interactive_model_cards import utils as ut


def quant_panel(sst_db):
    ''' Quantiative Panel Layout'''
    st.write("""<h1 style="font-size:20px"> Quantitative Examples</h1>""", unsafe_allow_html=True)

    quant_lcol, quant_rcol = st.columns([6, 6])

    all_metrics = {}
    with quant_lcol:
        for key in st.session_state['quant_ex']:
            #all_metrics[key] = st.session_state['quant_ex'][key]
            st.markdown("Overall Performance")
            chart = ut.visualize_metrics(sst_db.metrics['model'], max_width = 100)
            #st.markdown(f"{key}")
    st.altair_chart(chart)

       
    quant_rcol.write("Right Column")