import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import plotly.express as px
import plotly as ply
import plotly.graph_objs as go

app = dash.Dash(__name__)

df = pd.read_csv('../../data/rl/-seed50-ndrones1-mode2/qvalues.csv')

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

    html.H1(children="Histogram",),
    html.Div(id='istogramma-stati')
])


@app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
def update_heatmap(rows, derived_virtual_selected_rows):

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    STATES = [str([round(f, 3) for f in dff.iloc[:,1:13].iloc[i,:]]) for i in range(dff.iloc[:,1:13].shape[0])]
    fig = px.imshow(dff.iloc[:,13:-1],
                    labels=dict(x="actions", y="states", color="ntensity"),
                    x=dff.columns[13:-1],
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

    STATES = [str([round(f, 3) for f in dff.iloc[:,1:13].iloc[i,:]]) for i in range(dff.iloc[:,1:13].shape[0])]
    fig = px.histogram(dff.iloc[:,-1], 
                       labels=dict(x="states"),
                       x=STATES)
    fig.update_xaxes(showticklabels=False)

    return [dcc.Graph(id="2", figure=fig)]


if __name__ == '__main__':
    app.run_server(debug=True)





