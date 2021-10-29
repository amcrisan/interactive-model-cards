import streamlit as st
from interactive_model_cards import utils as ut


def quant_panel(sst_db):
    ''' Quantiative Panel Layout'''
    st.write("""<h1 style="font-size:20px"> Quantitative Examples</h1>""", unsafe_allow_html=True)

    quant_lcol, quant_rcol = st.columns([6, 6])

    with quant_lcol:
        st.markdown("Overall Performance")
        chart = ut.visualize_metrics(sst_db.metrics['model'], max_width = 100)
        st.altair_chart(chart)

        st.markdown("Custom Subset performance")

    with quant_rcol:
        st.write("Right Column")