# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

df_timeline = pd.read_csv('./data/imm_2015_2016_postprocessed.csv')

fig = go.Figure()
fig.add_bar(x=df_timeline.Date, y=df_timeline.Negative/(df_timeline.Total), name="% Negative")
fig.add_bar(x=df_timeline.Date, y=df_timeline.Positive/(df_timeline.Total), name="% Positive")
fig.add_trace(go.Scatter(x=[df_timeline.Date[0], df_timeline.Date[365]],y=[0.5,0.5],name="50% threshold"))
fig.update_layout(barmode="relative")
fig.update_layout(showlegend=True)
fig.show()

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
