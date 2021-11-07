
# --- Visualization ---
import altair as alt
import streamlit as st
import plotly.graph_objects as go
from streamlit_vega_lite import altair_component

# --- Data ---
import pandas as pd

def base_chart(df, linked_vis=False, max_width=150, col_val=None,min_size=100):
    ''' Visualize the model's performance across susbets of the data'''
    #Defining populations in the data
    pop_domain = ["Overall Performance","Custom Slice","User Custom Sentence","US Protected Class"]
    color_range = ["#5778a4", "#e49444", "#b8b0ac","#85b6b2"]
    
    #being chart
    base = alt.Chart(df)

    if linked_vis:
        selected = alt.selection_single(
            on="click", empty="none", fields=["name", "source"]
        )
        base = base.add_selection(selected)

        base = (
            base.mark_bar().encode(
                alt.X("metric_value", 
                    scale=alt.Scale(domain=(0, 1)), title=""
                ),
                alt.Y("displayName", title=""),
                alt.Column("metric_type", title=""),
                alt.StrokeWidth("size:N",
                    scale=alt.Scale(domain=[True,False],range=[0,1.25])
                ),
                alt.StrokeOpacity("size:N",
                    scale=alt.Scale(domain=[True,False],range=[0,1])
                    ),
                alt.Stroke("size:N",
                    scale=alt.Scale(domain=[True,False],range=["white","red"])
                ),
                alt.Fill("source",
                    scale = alt.Scale(domain = pop_domain,
                    range=color_range)),
                opacity=alt.condition(selected, alt.value(1), alt.value(0.5)),
                tooltip=["name", "metric_type", "metric_value"]
            )
            .properties(width=max_width)
        )
    else:
        #This is now depracted and should never occur
        base = (
            base.mark_bar()
            .encode(
                alt.X("metric_value", scale=alt.Scale(domain=(0, 1)), title=""),
                alt.Y(
                    "metric_type",
                    title="",
                    sort=["Overall Performance", "Your Sentences"],
                ),
                # alt.Row("metric_type",title=""),
                color=alt.value(col_val),
                tooltip=["name", "metric_type", "metric_value"],
            )
            .properties(width=max_width)
        )

    return base


@st.cache(allow_output_mutation=True)
def visualize_metrics(metrics, max_width=150, linked_vis=False, col_val="#1f77b4",min_size=1000):
    """
    Visualize the metrics of the model.
    """
    metric_df = pd.DataFrame()

    for key in metrics.keys():
        metric_types = []
        metric_values = []

        tmp = metrics[key]["metrics"]

        # get individual metrics
        for mt in tmp.keys():
            metric_types = metric_types + [mt]
            metric_values = metric_values + [tmp[mt]]

        name = [key] * len(metric_types)
        size = [metrics[key]["size"]] * len(metric_types)
        source = [metrics[key]["source"]] * len(metric_types)
        metric_df = metric_df.append(
            pd.DataFrame(
                {
                    "name": name,
                    "metric_type": metric_types,
                    "metric_value": metric_values,
                    "source": source,
                    "size" : [ True if x >= min_size else False for x in size]
                }
            )
        )


    #adding a human friendly display name (not RG's backend-name)
    tmp = [i.split("->") for i in metric_df['name']]
    metric_df['displayName']=[i.split("@")[0] for i in [j[0] if len(j)<=1 else j[1] for j in tmp ]]

    # generic metric chart
    base = base_chart(metric_df, linked_vis, col_val=col_val)

    # layered chart with line
    """
      # vertical line
    vertline = alt.Chart().mark_rule().encode(x="a:Q")
    metric_chart = (
        alt.layer(base, vertline,data=metric_df)
        .transform_calculate(a="0.5")
        .facet(
            alt.Column("metric_type", title=""))
        .configure_header(labelFontSize=12
        )
    )
    """

    return base


def vis_table(df, userInput=False):
    """ Visualize table data more effectively """
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                columnwidth=[400, 50, 50],
                cells=dict(
                    values=[df["sentence"], df["model label"], df["probability"]],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    return fig
