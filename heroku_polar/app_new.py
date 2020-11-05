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
], content)


content = html.Div(
    [
        html.H2('Analytics Dashboard Template', style=TEXT_STYLE),
        html.Hr(),
        content_first_row,
        content_second_row,
        content_third_row,
        content_fourth_row
    ],
    style=CONTENT_STYLE
)

#DASH CARDS - ONE PER THEME (WE HAVE 4 HERE - LGBTQ, POLITICS, IMMIGRATION, ENVIRONMENT)

content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_1', children=['Card Title 1'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4('Card Title 2', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Sample text.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Card Title 3', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Sample text.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Card Title 4', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Sample text.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]
        ),
        md=3
    )
])

# IN ONE CARD - I THEN PROCEED TO HAVE ONE FIGURE IN THE TOP ROW - THIS WILL BE THE YEAR TREND OF EACH THEME + 

content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_1'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_2'), md=4
        ),
    ]
)

# THEN WE HAVE 3 PLOTS IN A ROW REPRESENTING THE REACTIONS SPLIT - I.E. REACTIONS SPLIT BY SENTIMENT 

content_third_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_3'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_4'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_5'), md=4
        )
    ]
)

# THEN WE HAVE 3 PLOTS IN A ROW REPRESENTING THE REACTIONS CLASSIFICATION PER SENTIMENT 

content_fourth_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_6'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_7'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_8'), md=4
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
