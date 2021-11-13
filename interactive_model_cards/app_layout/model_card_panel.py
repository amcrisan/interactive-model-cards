import streamlit as st
import base64

def model_card_panel(model_card):
    """ Writing Model card in the sidebar"""
    # model card side panel
    for key in model_card.keys():
        item = model_card[key]
        
        st.sidebar.markdown(f"<h3>{model_card[key]['name']}</h3>", unsafe_allow_html=True)
       
        if "warning" in model_card[key].keys():
            #st.sidebar.error(model_card[key]["warning"])
            st.sidebar.markdown(
                f"""
                <span style='color:red;'>
                    <img src="data:image/png;base64,{base64.b64encode(open("./assets/img/warning.png", "rb").read()).decode()}"> {model_card[key]["warning"]}
                </span>
                """,
                unsafe_allow_html=True
            )

        n_short = len(model_card[key]['short'])
        if n_short == 1:
            st.sidebar.write(f"{model_card[key]['short'][0]}")
        else:
            for i in range(0,len(model_card[key]['short'])):
                st.sidebar.write(f"* {model_card[key]['short'][i]}")


        if "extended" in model_card[key].keys():
            with st.sidebar.expander(""):
                if len(model_card[key]["extended"]) > 1:
                    for detail in model_card[key]["extended"]:
                        st.markdown(f"* {detail}")
                else:
                    st.markdown(model_card[key]["extended"])

              
        else:
            st.sidebar.markdown("<hr class='line-one'>",unsafe_allow_html=True)