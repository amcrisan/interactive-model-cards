import pandas as pd
import altair as alt
import streamlit as st

def visualize_metrics(metrics,max_width = 150):
    """
    Visualize the metrics of the model.
    """    
    metric_df =  pd.DataFrame()

    for key in metrics.keys():
        metric_types = []
        metric_values  = []

        tmp = metrics[key]
        
        #get individual metrics
        for mt in tmp.keys():
            metric_types = metric_types + [mt]
            metric_values = metric_values + [tmp[mt]]

        name = [key] * len(metric_types)
        metric_df = metric_df.append(pd.DataFrame({'name':name, 'metric_type':metric_types, 'metric_value':metric_values}))

    #generic metric chart
    base = alt.Chart().mark_bar().encode(
        alt.X('metric_value',
            scale = alt.Scale(domain = (0,1)),
            title = ""),
        alt.Y('name',
            title = ""),
        tooltip = ['name','metric_type','metric_value']
    ).properties(
        #height = 20,
        width = max_width
    ).interactive()

    #vertical line
    vertline = alt.Chart().mark_rule().encode(
        x='a:Q'
    )


    #layered chart with line
    metric_chart = alt.layer(
        base,vertline,
        data=metric_df
    ).transform_calculate(
        a="0.5"
    ).facet(
        alt.Column("metric_type",
        title=""),
    ).configure_header(
        labelFontSize = 12
    )

    return(metric_chart)

