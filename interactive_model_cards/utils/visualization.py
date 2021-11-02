import pandas as pd
import altair as alt
import streamlit as st


from streamlit_vega_lite import altair_component


def base_chart(df,linked_vis=False,max_width=150,col_val=None):
    base = alt.Chart(df)

    if linked_vis:
        selected = alt.selection_single(on="click", empty="none",fields =['name'])
        base = base.add_selection(selected)   

        base = base.mark_bar(
            ).encode(
                alt.X("metric_value", scale=alt.Scale(domain=(0, 1)), title=""),
                alt.Y("name", title=""),
                alt.Column("metric_type",title=""),
                #alt.Row("source",title=""),
                color="source",
                #color=alt.condition(selected, alt.value("black"), "source"),
                opacity = alt.condition(selected,alt.value(1),alt.value(0.5)),
                tooltip=["name", "metric_type", "metric_value"]
            ).properties(
                width=150
            )
    else:
         base = base.mark_bar(
            ).encode(
                alt.X("metric_value", scale=alt.Scale(domain=(0, 1)), title=""),
                alt.Y("name", title=""),
                alt.Column("metric_type",title=""),
                color=alt.value(col_val),
                tooltip=["name", "metric_type", "metric_value"]
            ).properties(
                width=150 
            )

    return base
    

@st.cache(allow_output_mutation=True)
def visualize_metrics(metrics, max_width=150,linked_vis = False,col_val=None):
    """
    Visualize the metrics of the model.
    """
    metric_df = pd.DataFrame()

    for key in metrics.keys():
        metric_types = []
        metric_values = []

        tmp = metrics[key]['metrics']

        # get individual metrics
        for mt in tmp.keys():
            metric_types = metric_types + [mt]
            metric_values = metric_values + [tmp[mt]]

        name = [key] * len(metric_types)
        source = [metrics[key]['source']] * len(metric_types)
        metric_df = metric_df.append(
            pd.DataFrame(
                {
                    "name": name,
                    "metric_type": metric_types,
                    "metric_value": metric_values,
                    "source" : source
                }
            )
        )

    # generic metric chart

    base = base_chart(metric_df,linked_vis,col_val = col_val)
    #train= base_chart(metric_df[metric_df['name']=="sst(split=train, version=1.0.0)"],linked_vis=True)
    #test= base_chart(metric_df[metric_df['name']=="sst(split=test, version=1.0.0)"],linked_vis=True)

    #base  = alt.vconcat(train,test)

 
    # layered chart with line
    '''
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
    '''


    return base
