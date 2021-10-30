"""
 placeholder for all streamlit style hacks
"""
import streamlit as st


def init_style():
    return st.write(
        """
    <style>
    /* Side Bar */
    .css-1outpf7 {
        background-color:rgb(246 240 240);
        width:30rem;
        padding:10px 10px 10px 10px;
    }
    /* Main Panel*/
    .css-18e3th9 {
        padding:10px 10px 10px 10px;
    }
    .css-1ubw6au:last-child{
        background-color:lightblue;
    }

    /* Model Panels : element-container */
    .element-container{
            border-style:none
    }

    /* Radio Button Direction*/
    div.row-widget.stRadio > div{flex-direction:row;}

    /* Expander Boz*/
    .streamlit-expander {
        border-width: 0px;
        border-bottom: 1px solid #A29C9B;
        border-radius: 0px;
    }

    .streamlit-expanderHeader {
        font-style: italic;
        font-weight :600;
        padding-top:0px;
        padding-left: 0px;
        color:#A29C9B

    /* Section Headers */
    .sectionHeader {
        font-size:10px;
    }

    /* text input*/
    .st-e5 {
        background-color:lightblue;
    }
    </style>
""",
        unsafe_allow_html=True,
    )
