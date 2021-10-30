import streamlit as st


def model_card_panel(model_card):
    """ Writing Model card in the sidebar"""
    # model card side panel
    for key in model_card.keys():
        item = model_card[key]
        st.sidebar.markdown(f"**{model_card[key]['name']}**")
        st.sidebar.write(f"{model_card[key]['short'][0]}")

        with st.sidebar.expander("more details"):
            for detail in model_card[key]["short"]:
                st.markdown(f"* {detail}")
