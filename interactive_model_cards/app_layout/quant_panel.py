import streamlit as st
from interactive_model_cards import utils as ut
from streamlit_vega_lite import altair_component


def quant_panel(sst_db):
    """ Quantiative Panel Layout"""
    st.write(
        """<h1 style="font-size:20px"> Quantitative Examples</h1>""",
        unsafe_allow_html=True,
    )

    quant_lcol, quant_rcol = st.columns([8, 4])

    all_metrics = {}
    with quant_lcol:
        for key in st.session_state["quant_ex"]:
            tmp = st.session_state["quant_ex"][key]

            if tmp is not None:
                for iKey in tmp.keys():
                    all_metrics[iKey] = {}
                    all_metrics[iKey]['metrics'] = tmp[iKey]
                    all_metrics[iKey]['source'] = key 

        #st.write(all_metrics)
        chart = ut.visualize_metrics(all_metrics, max_width=100,linked_vis=True)
        event_dict = altair_component(altair_chart=chart)
        print(event_dict)
       
        #st.altair_chart(chart)

    quant_rcol.write("Right Column")
    quant_rcol.write(event_dict)

            # all_metrics[key] = st.session_state['quant_ex'][key]
            #st.markdown("Overall Performance")
            #chart = ut.visualize_metrics(sst_db.metrics["model"], max_width=100)
            #chart_title.markdown(f"{key}")
            #chart = ut.visualize_metrics(st.session_state["quant_ex"][key], max_width=100)
            #metric_chart.altair_chart(chart)
