"""
A simple app demonstrating how to dynamically render tab content containing
dcc.Graph components to ensure graphs get sized correctly. We also show how
dcc.Store can be used to cache the results of an expensive graph generation
process so that switching tabs is fast.
"""
import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.H1("Portraying OnLine hARm (POLAR)",style={'text-align':'center'}),
        html.Hr(),
        dbc.Button(
            "Click here to generate graphs",
            color="primary",
            block=True,
            id="button",
            className="mb-3",
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Immigration", tab_id="imm"),
                dbc.Tab(label="Politics", tab_id="politics"),
                dbc.Tab(label="LGBTQ+", tab_id="lgbtq"),
                dbc.Tab(label="Environment", tab_id="environment"),
            ],style={'font-weight': 'bold'},
            id="tabs",
            active_tab="imm",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        ################## IF CONDITIONS STARTING WITH IMMIGRATION
        if active_tab == "imm":
            return html.Div(
                [
                    dbc.Row(dbc.Col(html.Div("Number of tweets over 12 months split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["imm_2015"])),
                            dbc.Col(dcc.Graph(figure=data["imm_2019"])),
                         ],
                            no_gutters=True,
                            ),
                    
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to tweets over 12 months - split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["imm_2015_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["imm_2019_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to positive, negative and neutral tweets over 12 months - split by type", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["imm_2015_pos_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["imm_2019_pos_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["imm_2015_neg_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["imm_2019_neg_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["imm_2015_neut_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["imm_2019_neut_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                ]
            )
        ############################################ NOW FOR POLITICS ######################
        elif active_tab == "politics":
            return html.Div(
                [
                    dbc.Row(dbc.Col(html.Div("Number of tweets over 12 months split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["pol_2015"])),
                            dbc.Col(dcc.Graph(figure=data["pol_2019"])),
                         ],
                            no_gutters=True,
                            ),
                    
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to tweets over 12 months - split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["pol_2015_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["pol_2019_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to positive, negative and neutral tweets over 12 months - split by type", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["pol_2015_pos_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["pol_2019_pos_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["pol_2015_neg_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["pol_2019_neg_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["pol_2015_neut_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["pol_2019_neut_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                ]
            )
        ############################################ NOW FOR LGBTQ ######################
        elif active_tab == "lgbtq":
            return html.Div(
                [
                    dbc.Row(dbc.Col(html.Div("Number of tweets over 12 months split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["lgbt_2015"])),
                            dbc.Col(dcc.Graph(figure=data["lgbt_2019"])),
                         ],
                            no_gutters=True,
                            ),
                    
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to tweets over 12 months - split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["lgbt_2015_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["lgbt_2019_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to positive, negative and neutral tweets over 12 months - split by type", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["lgbt_2015_pos_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["lgbt_2019_pos_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["lgbt_2015_neg_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["lgbt_2019_neg_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["lgbt_2015_neut_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["lgbt_2019_neut_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                ]
            )
        
        ############################################ NOW FOR ENVIRONMENT ##########################################################
        elif active_tab == "environment":
             return html.Div(
                [
                    dbc.Row(dbc.Col(html.Div("Number of tweets over 12 months split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["env_2015"])),
                            dbc.Col(dcc.Graph(figure=data["env_2019"])),
                         ],
                            no_gutters=True,
                            ),
                    
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to tweets over 12 months - split by sentiment", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["env_2015_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["env_2019_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(dbc.Col(html.Div("Total number of reactions to positive, negative and neutral tweets over 12 months - split by type", 
                                             style={'color': 'black', 'fontSize': 20,'font-family':'Calibri', 'font-weight': 'bold'}))
                           ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["env_2015_pos_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["env_2019_pos_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["env_2015_neg_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["env_2019_neg_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=data["env_2015_neut_reactions"])),
                            dbc.Col(dcc.Graph(figure=data["env_2019_neut_reactions"])),
                         ],
                            no_gutters=True,
                            ),
                ]
            )
            
    return "No tab selected"

@app.callback(Output("store", "data"), [Input("button", "n_clicks")])
def generate_graphs(n):
    """
    This callback generates three simple graphs from random data.
    """
    if not n:
        # generate empty graphs when app loads
        return {k: go.Figure(data=[]) for k in ["imm_2015", "imm_2019","imm_2015_reactions","imm_2019_reactions","imm_2015_pos_reactions","imm_2019_pos_reactions","imm_2015_neg_reactions","imm_2019_neg_reactions","imm_2015_neut_reactions","imm_2019_neut_reactions",
                                                "pol_2015", "pol_2019","pol_2015_reactions","pol_2019_reactions","pol_2015_pos_reactions","pol_2019_pos_reactions","pol_2015_neg_reactions","pol_2019_neg_reactions","pol_2015_neut_reactions","pol_2019_neut_reactions",
                                                "lgbt_2015", "lgbt_2019","lgbt_2015_reactions","lgbt_2019_reactions","lgbt_2015_pos_reactions","lgbt_2019_pos_reactions","lgbt_2015_neg_reactions","lgbt_2019_neg_reactions","lgbt_2015_neut_reactions","lgbt_2019_neut_reactions",
                                                "env_2015", "env_2019","env_2015_reactions","env_2019_reactions","env_2015_pos_reactions","env_2019_pos_reactions","env_2015_neg_reactions","env_2019_neg_reactions","env_2015_neut_reactions","env_2019_neut_reactions"]}

    ################################################################################################################################    
    ###################################################### START OF PLOT CREATION ####################################################
    ################################################################################################################################    
    
    ############################################################################################################
    ########################################## IMMIGRATION #####################################################
    ############################################################################################################
    
    #################### importing immigration postprocessed dataframe from 2015/2016 and 2019/2020
    imm_pp_pre = pd.read_csv('./data_postprocessed/imm_2015_2016_postprocessed.csv')
    imm_pp_post = pd.read_csv('./data_postprocessed/imm_2019_2020_postprocessed.csv')
    
    #################### Calculating max to use for y-axis range
    max_range_yaxis = max(imm_pp_pre.Total.max(),imm_pp_post.Total.max())
    
    
    ################################################################################################## 2015-2016
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral
    
    imm_2015 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    imm_2015.add_trace(go.Scatter(x=imm_pp_pre.Date, y=imm_pp_pre.Neutral, mode='lines',name = "Neutral",
                                  stackgroup='one'), row=1,col=1) 
    imm_2015.add_trace(go.Scatter(x=imm_pp_pre.Date, y=imm_pp_pre.Negative, mode='lines',name = "Negative",
                                  stackgroup='one'),row=1,col=1)
    imm_2015.add_trace(go.Scatter(x=imm_pp_pre.Date, y=imm_pp_pre.Positive, mode='lines',name = "Positive",
                                  stackgroup='one'),row=1,col=1)    
    imm_2015.add_trace(go.Scatter(x=imm_pp_pre.Date, y=imm_pp_pre.Percentage_neg, fill='tozeroy', 
                                  name="% Negative",fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)

    imm_2015.update_layout(showlegend=True, 
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    imm_2015.update_yaxes(title_text="# of tweets", row=1, col=1, range=[0, max_range_yaxis+100])
    imm_2015.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    imm_2015.layout.font.family = 'Calibri'

    imm_2015.show()
                           
    #################################################################################################### 2019-2020                       
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral    
    imm_2019 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    imm_2019.add_trace(go.Scatter(x=imm_pp_post.Date, y=imm_pp_post.Neutral, mode='lines',name = "Neutral",
                                  stackgroup='one'), row=1,col=1) 
    imm_2019.add_trace(go.Scatter(x=imm_pp_post.Date, y=imm_pp_post.Negative, mode='lines',name = "Negative",
                                  stackgroup='one'),row=1,col=1)
    imm_2019.add_trace(go.Scatter(x=imm_pp_post.Date, y=imm_pp_post.Positive, mode='lines',name = "Positive",
                                  stackgroup='one'),row=1,col=1)    
    imm_2019.add_trace(go.Scatter(x=imm_pp_post.Date, y=imm_pp_post.Percentage_neg, fill='tozeroy', 
                                  name="% Negative",fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)

    imm_2019.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    imm_2019.update_yaxes(title_text="# of tweets", row=1, col=1,range=[0, max_range_yaxis+100])
    imm_2019.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    imm_2019.layout.font.family = 'Calibri'

    imm_2019.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2015-2016
    imm_2015_reac = pd.read_csv('./data_postprocessed/imm_2015_2016_reactions_hor.csv')
    #------------------------------- TOTAL REACTIONS
    
    imm_2015_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    imm_2015_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    imm_2015_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    imm_2015_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    imm_2015_reac_total = imm_2015_reac.Total_Neutral+imm_2015_reac.Total_Negative + imm_2015_reac.Total_Positive
    imm_2015_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=100*imm_2015_reac.Total_Negative/imm_2015_reac_total, fill='tozeroy', 
                                  name="% Negative",fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    imm_2015_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    imm_2015_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    imm_2015_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    
    imm_2015_reactions.layout.font.family = 'Calibri'
    imm_2015_reactions.show()
    
    #------------------------------- POSITIVE
    imm_2015_pos_reactions = go.Figure()
    imm_2015_pos_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    imm_2015_pos_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    imm_2015_pos_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Positive_nretweets,
                                            mode='lines',name = "Retweets",stackgroup='one'))
    imm_2015_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    imm_2015_pos_reactions.layout.font.family = 'Calibri'
    imm_2015_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    imm_2015_neg_reactions = go.Figure()
    imm_2015_neg_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    imm_2015_neg_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    imm_2015_neg_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    imm_2015_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    imm_2015_neg_reactions.layout.font.family = 'Calibri'
    imm_2015_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    imm_2015_neut_reactions = go.Figure()
    imm_2015_neut_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    imm_2015_neut_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    imm_2015_neut_reactions.add_trace(go.Scatter(x=imm_2015_reac.Date, y=imm_2015_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    imm_2015_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    imm_2015_neut_reactions.layout.font.family = 'Calibri'
    imm_2015_neut_reactions.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2019-2020
    
    imm_2019_reac = pd.read_csv('./data_postprocessed/imm_2019_2020_reactions_hor.csv')
    
    #------------------------------- TOTAL REACTIONS
    imm_2019_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.7,0.2])
    imm_2019_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    imm_2019_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    imm_2019_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    imm_2019_reac_total = imm_2019_reac.Total_Neutral+imm_2019_reac.Total_Negative + imm_2019_reac.Total_Positive
    imm_2019_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=100*imm_2019_reac.Total_Negative/imm_2019_reac_total, fill='tozeroy', 
                                  name="% Negative",fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    imm_2019_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    imm_2019_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    imm_2019_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    
    imm_2019_reactions.layout.font.family = 'Calibri'
    imm_2019_reactions.show()
    
    #------------------------------- POSITIVE
    
    imm_2019_pos_reactions = go.Figure()
    imm_2019_pos_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    imm_2019_pos_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    imm_2019_pos_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    imm_2019_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    imm_2019_pos_reactions.layout.font.family = 'Calibri'
    imm_2019_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    imm_2019_neg_reactions = go.Figure()
    imm_2019_neg_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    imm_2019_neg_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    imm_2019_neg_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    imm_2019_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    imm_2019_neg_reactions.layout.font.family = 'Calibri'
    imm_2019_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    imm_2019_neut_reactions = go.Figure()
    imm_2019_neut_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    imm_2019_neut_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    imm_2019_neut_reactions.add_trace(go.Scatter(x=imm_2019_reac.Date, y=imm_2019_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    imm_2019_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    imm_2019_neut_reactions.layout.font.family = 'Calibri'
    imm_2019_neut_reactions.show()
    
    ############################################################################################################
    ########################################## POLITICS ########################################################
    ############################################################################################################
    
    #################### importing politics postprocessed dataframe from 2015/2016 and 2019/2020
    pol_pp_pre = pd.read_csv('./data_postprocessed/politics_2015_2016_postprocessed.csv')
    pol_pp_post = pd.read_csv('./data_postprocessed/politics_2019_2020_postprocessed.csv')
    
    #################### Calculating max to use for y-axis range
    max_range_yaxis = max(pol_pp_pre.Total.max(),pol_pp_post.Total.max())
    
    ################################################################################################## 2015-2016
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral
    pol_2015 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.7,0.2])

    pol_2015.add_trace(go.Scatter(x=pol_pp_pre.Date, y=pol_pp_pre.Neutral, mode='lines', name = "Neutral",
                                  stackgroup='one'), row=1, col=1)
    pol_2015.add_trace(go.Scatter(x=pol_pp_pre.Date, y=pol_pp_pre.Negative, mode='lines',name = "Negative",
                                  stackgroup='one'), row=1,col=1)
    pol_2015.add_trace(go.Scatter(x=pol_pp_pre.Date, y=pol_pp_pre.Positive, mode='lines',name = "Positive",
                                  stackgroup='one'), row=1,col=1)
    pol_2015.add_trace(go.Scatter(x=pol_pp_pre.Date, y=pol_pp_pre.Percentage_neg, fill='tozeroy', 
                                  fillcolor='rgba(255,0,0,0.4)', line_color='red',name="% Negative"),  row=2, col=1)

    pol_2015.update_layout(showlegend=True, 
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    pol_2015.update_yaxes(title_text="# of tweets", row=1, col=1, range=[0, max_range_yaxis+100])
    pol_2015.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])
    pol_2015.layout.font.family = 'Calibri'

    pol_2015.show()
                           
    ################################################################################################## 2019-2020                       
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral    
    pol_2019 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    pol_2019.add_trace(go.Scatter(x=pol_pp_post.Date, y=pol_pp_post.Neutral, mode='lines', name = "Neutral",
                                  stackgroup='one'), row=1, col=1)
    pol_2019.add_trace(go.Scatter(x=pol_pp_post.Date, y=pol_pp_post.Negative, mode='lines',name = "Negative",
                                  stackgroup='one'), row=1,col=1)
    pol_2019.add_trace(go.Scatter(x=pol_pp_post.Date, y=pol_pp_post.Positive, mode='lines',name = "Positive",
                                  stackgroup='one'), row=1,col=1)
    pol_2019.add_trace(go.Scatter(x=pol_pp_post.Date, y=pol_pp_post.Percentage_neg, fill='tozeroy', 
                                  fillcolor='rgba(255,0,0,0.4)',line_color='red',name="% Negative"),  row=2, col=1)

    pol_2019.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    pol_2019.update_yaxes(title_text="# of tweets", row=1, col=1,range=[0, max_range_yaxis+100])
    pol_2019.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    pol_2019.layout.font.family = 'Calibri'

    pol_2019.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2015-2016
    pol_2015_reac = pd.read_csv('./data_postprocessed/politics_2015_2016_reactions_hor.csv')
    
    #------------------------------- TOTAL REACTIONS
    pol_2015_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    pol_2015_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    pol_2015_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    pol_2015_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    pol_2015_reac_total = pol_2015_reac.Total_Neutral+pol_2015_reac.Total_Negative + pol_2015_reac.Total_Positive
    pol_2015_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=100*pol_2015_reac.Total_Negative/pol_2015_reac_total, fill='tozeroy', 
                                  name="% Negative",fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    pol_2015_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    pol_2015_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    pol_2015_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    pol_2015_reactions.layout.font.family = 'Calibri'
    pol_2015_reactions.show()
    
    #------------------------------- POSITIVE
    pol_2015_pos_reactions = go.Figure()
    pol_2015_pos_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    pol_2015_pos_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    pol_2015_pos_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    pol_2015_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    pol_2015_pos_reactions.layout.font.family = 'Calibri'
    pol_2015_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    pol_2015_neg_reactions = go.Figure()
    pol_2015_neg_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    pol_2015_neg_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    pol_2015_neg_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    pol_2015_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    pol_2015_neg_reactions.layout.font.family = 'Calibri'
    pol_2015_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    pol_2015_neut_reactions = go.Figure()
    pol_2015_neut_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    pol_2015_neut_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    pol_2015_neut_reactions.add_trace(go.Scatter(x=pol_2015_reac.Date, y=pol_2015_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    pol_2015_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    pol_2015_neut_reactions.layout.font.family = 'Calibri'
    pol_2015_neut_reactions.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2019-2020
    
    pol_2019_reac = pd.read_csv('./data_postprocessed/politics_2019_2020_reactions_hor.csv')
    #------------------------------- TOTAL REACTIONS
    pol_2019_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    pol_2019_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    pol_2019_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    pol_2019_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    pol_2019_reac_total = pol_2019_reac.Total_Neutral+pol_2019_reac.Total_Negative + pol_2019_reac.Total_Positive
    pol_2019_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=100*pol_2019_reac.Total_Negative/pol_2019_reac_total, fill='tozeroy', 
                                  name="% Negative",fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    pol_2019_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    pol_2019_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    pol_2019_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    pol_2019_reactions.layout.font.family = 'Calibri'
    pol_2019_reactions.show()
    #------------------------------- POSITIVE
    pol_2019_pos_reactions = go.Figure()
    pol_2019_pos_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    pol_2019_pos_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    pol_2019_pos_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    pol_2019_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    pol_2019_pos_reactions.layout.font.family = 'Calibri'
    pol_2019_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    pol_2019_neg_reactions = go.Figure()
    pol_2019_neg_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    pol_2019_neg_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    pol_2019_neg_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    pol_2019_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    pol_2019_neg_reactions.layout.font.family = 'Calibri'
    pol_2019_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    pol_2019_neut_reactions = go.Figure()
    pol_2019_neut_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    pol_2019_neut_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    pol_2019_neut_reactions.add_trace(go.Scatter(x=pol_2019_reac.Date, y=pol_2019_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    pol_2019_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    pol_2019_neut_reactions.layout.font.family = 'Calibri'
    pol_2019_neut_reactions.show()    
    
    ############################################################################################################
    ########################################## LGBTQ ########################################################
    ############################################################################################################
    
    #################### importing lgbtitics postprocessed dataframe from 2015/2016 and 2019/2020
    lgbt_pp_pre = pd.read_csv('./data_postprocessed/lgbtq_2015_2016_postprocessed.csv')
    lgbt_pp_post = pd.read_csv('./data_postprocessed/lgbtq_2019_2020_postprocessed.csv')
    
    #################### Calculating max to use for y-axis range
    max_range_yaxis = max(lgbt_pp_pre.Total.max(),lgbt_pp_post.Total.max())
    
    ################################################################################################## 2015-2016
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral
    lgbt_2015 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.7,0.2])

    lgbt_2015.add_trace(go.Scatter(x=lgbt_pp_pre.Date, y=lgbt_pp_pre.Neutral, mode='lines', 
                                   name = "Neutral",stackgroup='one'), row=1, col=1)
    lgbt_2015.add_trace(go.Scatter(x=lgbt_pp_pre.Date, y=lgbt_pp_pre.Negative, mode='lines',
                                   name = "Negative",stackgroup='one'), row=1,col=1)
    lgbt_2015.add_trace(go.Scatter(x=lgbt_pp_pre.Date, y=lgbt_pp_pre.Positive, mode='lines',
                                   name = "Positive",stackgroup='one'), row=1,col=1)
    lgbt_2015.add_trace(go.Scatter(x=lgbt_pp_pre.Date, y=lgbt_pp_pre.Percentage_neg, 
                                   fillcolor='rgba(255,0,0,0.4)', line_color='red',fill='tozeroy', name="% Negative"),  row=2, col=1)

    lgbt_2015.update_layout(showlegend=True, 
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    lgbt_2015.update_yaxes(title_text="# of tweets", row=1, col=1, range=[0, max_range_yaxis+100])
    lgbt_2015.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    lgbt_2015.layout.font.family = 'Calibri'

    lgbt_2015.show()
                           
    ################################################################################################## 2019-2020                       
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral    
    lgbt_2019 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    lgbt_2019.add_trace(go.Scatter(x=lgbt_pp_post.Date, y=lgbt_pp_post.Neutral, mode='lines', 
                                   name = "Neutral",stackgroup='one'), row=1, col=1)
    lgbt_2019.add_trace(go.Scatter(x=lgbt_pp_post.Date, y=lgbt_pp_post.Negative, mode='lines',
                                   name = "Negative",stackgroup='one'), row=1,col=1)
    lgbt_2019.add_trace(go.Scatter(x=lgbt_pp_post.Date, y=lgbt_pp_post.Positive, mode='lines',
                                   name = "Positive",stackgroup='one'), row=1,col=1)
    lgbt_2019.add_trace(go.Scatter(x=lgbt_pp_post.Date, y=lgbt_pp_post.Percentage_neg, 
                                   fillcolor='rgba(255,0,0,0.4)', line_color='red',fill='tozeroy', name="% Negative"),  row=2, col=1)

    lgbt_2019.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    lgbt_2019.update_yaxes(title_text="# of tweets", row=1, col=1,range=[0, max_range_yaxis+100])
    lgbt_2019.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    lgbt_2019.layout.font.family = 'Calibri'

    lgbt_2019.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2015-2016
    lgbt_2015_reac = pd.read_csv('./data_postprocessed/lgbtq_2015_2016_reactions_hor.csv')
    
    #------------------------------- TOTAL REACTIONS
    
    lgbt_2015_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    lgbt_2015_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    lgbt_2015_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    lgbt_2015_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    lgbt_2015_reac_total = lgbt_2015_reac.Total_Neutral+lgbt_2015_reac.Total_Negative + lgbt_2015_reac.Total_Positive
    lgbt_2015_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=100*lgbt_2015_reac.Total_Negative/lgbt_2015_reac_total,
                                             fill='tozeroy',name="% Negative",
                                             fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    lgbt_2015_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    lgbt_2015_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    lgbt_2015_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    lgbt_2015_reactions.layout.font.family = 'Calibri'
    lgbt_2015_reactions.show()
    
    #------------------------------- POSITIVE
    lgbt_2015_pos_reactions = go.Figure()
    lgbt_2015_pos_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    lgbt_2015_pos_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    lgbt_2015_pos_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    lgbt_2015_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    lgbt_2015_pos_reactions.layout.font.family = 'Calibri'
    lgbt_2015_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    lgbt_2015_neg_reactions = go.Figure()
    lgbt_2015_neg_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    lgbt_2015_neg_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    lgbt_2015_neg_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    lgbt_2015_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    lgbt_2015_neg_reactions.layout.font.family = 'Calibri'
    lgbt_2015_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    lgbt_2015_neut_reactions = go.Figure()
    lgbt_2015_neut_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    lgbt_2015_neut_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    lgbt_2015_neut_reactions.add_trace(go.Scatter(x=lgbt_2015_reac.Date, y=lgbt_2015_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    lgbt_2015_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    lgbt_2015_neut_reactions.layout.font.family = 'Calibri'
    lgbt_2015_neut_reactions.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2019-2020
    
    lgbt_2019_reac = pd.read_csv('./data_postprocessed/lgbtq_2019_2020_reactions_hor.csv')
    #------------------------------- TOTAL REACTIONS
    lgbt_2019_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    lgbt_2019_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    lgbt_2019_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    lgbt_2019_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    lgbt_2019_reac_total = lgbt_2019_reac.Total_Neutral+lgbt_2019_reac.Total_Negative + lgbt_2019_reac.Total_Positive
    lgbt_2019_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=100*lgbt_2019_reac.Total_Negative/lgbt_2019_reac_total,
                                             fill='tozeroy',name="% Negative",
                                             fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    lgbt_2019_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    lgbt_2019_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    lgbt_2019_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    lgbt_2019_reactions.layout.font.family = 'Calibri'
    lgbt_2019_reactions.show()
    #------------------------------- POSITIVE
    lgbt_2019_pos_reactions = go.Figure()
    lgbt_2019_pos_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    lgbt_2019_pos_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    lgbt_2019_pos_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    lgbt_2019_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    lgbt_2019_pos_reactions.layout.font.family = 'Calibri'
    lgbt_2019_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    lgbt_2019_neg_reactions = go.Figure()
    lgbt_2019_neg_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    lgbt_2019_neg_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    lgbt_2019_neg_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    lgbt_2019_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    lgbt_2019_neg_reactions.layout.font.family = 'Calibri'
    lgbt_2019_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    lgbt_2019_neut_reactions = go.Figure()
    lgbt_2019_neut_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    lgbt_2019_neut_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    lgbt_2019_neut_reactions.add_trace(go.Scatter(x=lgbt_2019_reac.Date, y=lgbt_2019_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    lgbt_2019_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    lgbt_2019_neut_reactions.layout.font.family = 'Calibri'
    lgbt_2019_neut_reactions.show()
    
    ############################################################################################################
    ########################################## ENVIRONMENT #####################################################
    ############################################################################################################
    
    #################### importing envigration postprocessed dataframe from 2015/2016 and 2019/2020
    env_pp_pre = pd.read_csv('./data_postprocessed/env_2015_2016_postprocessed.csv')
    env_pp_post = pd.read_csv('./data_postprocessed/env_2019_2020_postprocessed.csv')
    
    #################### Calculating max to use for y-axis range
    max_range_yaxis = max(env_pp_pre.Total.max(),env_pp_post.Total.max())
    
    
    ################################################################################################## 2015-2016
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral
    env_2015 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.7,0.2])

    env_2015.add_trace(go.Scatter(x=env_pp_pre.Date, y=env_pp_pre.Neutral, mode='lines', 
                                  name = "Neutral",stackgroup='one'), row=1, col=1)
    env_2015.add_trace(go.Scatter(x=env_pp_pre.Date, y=env_pp_pre.Negative, mode='lines',
                                  name = "Negative",stackgroup='one'), row=1,col=1)
    env_2015.add_trace(go.Scatter(x=env_pp_pre.Date, y=env_pp_pre.Positive, mode='lines',
                                  name = "Positive",stackgroup='one'), row=1,col=1)
    env_2015.add_trace(go.Scatter(x=env_pp_pre.Date, y=env_pp_pre.Percentage_neg, fill='tozeroy',
                                  fillcolor='rgba(255,0,0,0.4)', line_color='red',name="% Negative"),  row=2, col=1)

    env_2015.update_layout(showlegend=True, 
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    env_2015.update_yaxes(title_text="# of tweets", row=1, col=1, range=[0, max_range_yaxis+100])
    env_2015.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    env_2015.layout.font.family = 'Calibri'

    env_2015.show()
                           
    #################################################################################################### 2019-2020                       
    #----------------------# Tweets split by sentiment: # positive, # negative and # neutral    
    env_2019 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    
    env_2019.add_trace(go.Scatter(x=env_pp_post.Date, y=env_pp_post.Neutral, mode='lines', 
                                  name = "Neutral",stackgroup='one'), row=1, col=1)
    env_2019.add_trace(go.Scatter(x=env_pp_post.Date, y=env_pp_post.Negative, mode='lines',
                                  name = "Negative",stackgroup='one'), row=1,col=1)
    env_2019.add_trace(go.Scatter(x=env_pp_post.Date, y=env_pp_post.Positive, mode='lines',
                                  name = "Positive",stackgroup='one'), row=1,col=1)
    env_2019.add_trace(go.Scatter(x=env_pp_post.Date, y=env_pp_post.Percentage_neg, fill='tozeroy',
                                  fillcolor='rgba(255,0,0,0.4)', line_color='red',name="% Negative"),  row=2, col=1)

    env_2019.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    env_2019.update_yaxes(title_text="# of tweets", row=1, col=1,range=[0, max_range_yaxis+100])
    env_2019.update_yaxes(title_text="% neg", row=2, col=1,range=[0, 100])

    env_2019.layout.font.family = 'Calibri'

    env_2019.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2015-2016
    env_2015_reac = pd.read_csv('./data_postprocessed/env_2015_2016_reactions_hor.csv')
    
    #------------------------------- TOTAL REACTIONS
    env_2015_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    env_2015_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    env_2015_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    env_2015_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    env_2015_reac_total = env_2015_reac.Total_Neutral+env_2015_reac.Total_Negative + env_2015_reac.Total_Positive
    env_2015_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=100*env_2015_reac.Total_Negative/env_2015_reac_total,
                                             fill='tozeroy',name="% Negative",
                                             fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    env_2015_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    env_2015_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    env_2015_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    env_2015_reactions.layout.font.family = 'Calibri'
    env_2015_reactions.show()
    
    #------------------------------- POSITIVE
    env_2015_pos_reactions = go.Figure()
    env_2015_pos_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    env_2015_pos_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    env_2015_pos_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    env_2015_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    env_2015_pos_reactions.layout.font.family = 'Calibri'
    env_2015_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    env_2015_neg_reactions = go.Figure()
    env_2015_neg_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    env_2015_neg_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    env_2015_neg_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    env_2015_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    env_2015_neg_reactions.layout.font.family = 'Calibri'
    env_2015_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    env_2015_neut_reactions = go.Figure()
    env_2015_neut_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    env_2015_neut_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    env_2015_neut_reactions.add_trace(go.Scatter(x=env_2015_reac.Date, y=env_2015_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    env_2015_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2015 to Sep 2016',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    env_2015_neut_reactions.layout.font.family = 'Calibri'
    env_2015_neut_reactions.show()
    
    ####################################################################### Positive, negative, neutral tweets split by reaction type 
    ####################################################################################################################### 2019-2020
    
    env_2019_reac = pd.read_csv('./data_postprocessed/env_2019_2020_reactions_hor.csv')
    
    #------------------------------- TOTAL REACTIONS
    env_2019_reactions = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, row_heights=[0.7,0.2])
    env_2019_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Total_Neutral,
                                            mode='lines', name = "Neutral",stackgroup='one'), row=1,col=1)
    env_2019_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Total_Negative,
                                            mode='lines',name = "Negative",stackgroup='one'), row=1,col=1)
    env_2019_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Total_Positive, 
                                            mode='lines',name = "Positive",stackgroup='one'), row=1,col=1)
    env_2019_reac_total = env_2019_reac.Total_Neutral+env_2019_reac.Total_Negative + env_2019_reac.Total_Positive
    env_2019_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=100*env_2019_reac.Total_Negative/env_2019_reac_total,
                                             fill='tozeroy',name="% Negative",
                                             fillcolor='rgba(255,0,0,0.4)', line_color='red'),  row=2, col=1)
    
    env_2019_reactions.update_layout(showlegend=True,
           title=dict(
                text='Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))

    env_2019_reactions.update_yaxes(title_text="# of reactions", row=1, col=1)
    env_2019_reactions.update_yaxes(title_text="% (-) tweets", row=2, col=1,range=[0, 100])
    env_2019_reactions.layout.font.family = 'Calibri'
    env_2019_reactions.show()
    
    #------------------------------- POSITIVE
    
    env_2019_pos_reactions = go.Figure()
    env_2019_pos_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Positive_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    env_2019_pos_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Positive_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    env_2019_pos_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Positive_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    env_2019_pos_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (+) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    env_2019_pos_reactions.layout.font.family = 'Calibri'
    env_2019_pos_reactions.show()
     
    #-------------------------------- NEGATIVE
        
    env_2019_neg_reactions = go.Figure()
    env_2019_neg_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Negative_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    env_2019_neg_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Negative_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    env_2019_neg_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Negative_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    env_2019_neg_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (-) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    env_2019_neg_reactions.layout.font.family = 'Calibri'
    env_2019_neg_reactions.show()
     
    #-------------------------------- NEUTRAL
        
    env_2019_neut_reactions = go.Figure()
    env_2019_neut_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Neutral_nlikes, 
                                            mode='lines',name = "Likes",stackgroup='one'))
    env_2019_neut_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Neutral_nreplies, 
                                            mode='lines',name = "Replies",stackgroup='one'))
    env_2019_neut_reactions.add_trace(go.Scatter(x=env_2019_reac.Date, y=env_2019_reac.Neutral_nretweets, 
                                            mode='lines', name = "Retweets",stackgroup='one'))
    env_2019_neut_reactions.update_layout(yaxis_title="# of reactions",showlegend=True,
           title=dict(
                text='Reactions to (/) tweets - Oct 2019 to Sep 2020',xanchor= "center",x=0.5,
                font=dict(size=20,color='#000000')),
           legend=dict(orientation="h", yanchor="bottom",y=1.02,xanchor="right",x=1))
    env_2019_neut_reactions.layout.font.family = 'Calibri'
    env_2019_neut_reactions.show()
    
    ################################################################################################################################    
    ###################################################### END OF PLOT CREATION ####################################################
    ################################################################################################################################

    # save figures in a dictionary for sending to the dcc.Store
    return {"imm_2015": imm_2015, "imm_2019": imm_2019,
            "imm_2015_reactions" : imm_2015_reactions,
            "imm_2019_reactions" : imm_2019_reactions,
            "imm_2015_pos_reactions": imm_2015_pos_reactions,  
            "imm_2015_neg_reactions": imm_2015_neg_reactions,  
            "imm_2015_neut_reactions": imm_2015_neut_reactions,
            "imm_2019_pos_reactions": imm_2019_pos_reactions,  
            "imm_2019_neg_reactions": imm_2019_neg_reactions,  
            "imm_2019_neut_reactions": imm_2019_neut_reactions,
            "pol_2015": pol_2015, "pol_2019": pol_2019,
            "pol_2015_reactions" : pol_2015_reactions,
            "pol_2019_reactions" : pol_2019_reactions,
            "pol_2015_pos_reactions": pol_2015_pos_reactions,  
            "pol_2015_neg_reactions": pol_2015_neg_reactions,  
            "pol_2015_neut_reactions": pol_2015_neut_reactions,
            "pol_2019_pos_reactions": pol_2019_pos_reactions,  
            "pol_2019_neg_reactions": pol_2019_neg_reactions,  
            "pol_2019_neut_reactions": pol_2019_neut_reactions,
            "lgbt_2015": lgbt_2015, "lgbt_2019": lgbt_2019,
            "lgbt_2015_reactions" : lgbt_2015_reactions,
            "lgbt_2019_reactions" : lgbt_2019_reactions,
            "lgbt_2015_pos_reactions": lgbt_2015_pos_reactions,  
            "lgbt_2015_neg_reactions": lgbt_2015_neg_reactions,  
            "lgbt_2015_neut_reactions": lgbt_2015_neut_reactions,
            "lgbt_2019_pos_reactions": lgbt_2019_pos_reactions,  
            "lgbt_2019_neg_reactions": lgbt_2019_neg_reactions,  
            "lgbt_2019_neut_reactions": lgbt_2019_neut_reactions,
            "env_2015": env_2015, "env_2019": env_2019,
            "env_2015_reactions" : env_2015_reactions,
            "env_2019_reactions" : env_2019_reactions,
            "env_2015_pos_reactions": env_2015_pos_reactions,  
            "env_2015_neg_reactions": env_2015_neg_reactions,  
            "env_2015_neut_reactions": env_2015_neut_reactions,
            "env_2019_pos_reactions": env_2019_pos_reactions,  
            "env_2019_neg_reactions": env_2019_neg_reactions,  
            "env_2019_neut_reactions": env_2019_neut_reactions}


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)