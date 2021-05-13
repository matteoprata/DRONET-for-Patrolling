import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly

import plotly.express as px

N_ACTIONS = 5
N_FEATURES = 12

app = dash.Dash(__name__)

df = pd.read_csv('data/rl/-seed50-ndrones1-mode2/qvalues.csv', index_col=0).sort_values(by="count", ascending=False)[:1000]

app.layout = html.Div([
    html.H1(children="Q-Table",),

    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        # editable=True,
        # filter_action="native",
        sort_action="native",
        sort_mode="multi",
        # column_selectable="single",
        # row_selectable="multi",
        # row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    ),

    html.H1(children="Heatmap",),
    html.Div(id='datatable-interactivity-container'),

    # html.H1(children="Histogram",),
    # html.Div(id='istogramma-stati')
])


@app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
def update_heatmap(rows, derived_virtual_selected_rows):

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    STATES = [str([round(f, 3) for f in dff.iloc[:,:N_FEATURES].iloc[i,:]]) for i in range(dff.iloc[:,:N_FEATURES].shape[0])]
    fig = px.imshow(dff.iloc[:,N_FEATURES:-1],
                    labels=dict(x="actions", y="states", color="ntensity"),
                    x=dff.columns[N_FEATURES:-1],
                    y=STATES
                    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=500,
    )

    fig.update_layout(
    autosize=False,
    width=2000,
    height=800,)

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(side="top")

    return [dcc.Graph(id="1", figure=fig)]


@app.callback(
    Output('istogramma-stati', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
def states_histogram(rows, derived_virtual_selected_rows):

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    STATES = [str([f for f in dff.iloc[:,:N_FEATURES].iloc[i,:]]) for i in range(dff.iloc[:,:N_FEATURES].shape[0])]
    weights = dff.iloc[:, -1]

    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # ax.plot([1, 2, 3, 4])

    ax.hist(STATES, weights=weights)

    # fig = plt.gcf()
    plt.show()
    return [dcc.Graph(id="2", figure=fig)]


if __name__ == '__main__':
    app.run_server(debug=True)





