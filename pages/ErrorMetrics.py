import pandas as pd
import numpy as np

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_table
from dateutil.relativedelta import relativedelta
from dash import callback
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# MAPE metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))*100

def WAPE(y_true, y_pred):
    return (y_true - y_pred).abs().sum()*100 / y_true.abs().sum()

# Function for styling the table
import colorlover
def discrete_background_color_bins(df, n_bins=7):
    
    bounds = [i * (1.0 / n_bins) for i in range(n_bins+1)]
    data = df.select_dtypes('number')
    df_max = data.max().max()
    df_min = data.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins+4)]['div']['RdYlGn'][2:-2][len(bounds) - i - 1]
        color = 'black'

        for column in data.columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
            styles.append({
                    'if': {
                        'filter_query': '{{{}}} is blank'.format(column),
                        'column_id': column
                    },
                    'backgroundColor': '#FFFFFF',
                    'color': 'white'
                })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))



# Importing error matrix data
df_sales = pd.read_csv('sales_error_tracker.csv')
df_som = pd.read_csv('som_error_tracker.csv')

# Rounding to 2 decimal places
df_sales = df_sales.round(decimals = 2)
df_som = df_som.round(decimals = 2)

# df_sales.columns = ['Manufacturer', 'Channel', 'Training Upto', 'October 2021',
#        'November 2021', 'December 2021', 'January 2022', 'February 2022',
#        'March 2022', 'April 2022', 'May 2022', 'June 2022', 'July 2022',
#        'August 2022', 'Overall WAPE']
# df_som.columns = ['Manufacturer', 'Channel', 'Training Upto', 'October 2021',
#        'November 2021', 'December 2021', 'January 2022', 'February 2022',
#        'March 2022', 'April 2022', 'May 2022', 'June 2022', 'July 2022',
#        'August 2022', 'Overall MAE']

# Select display columns in UI
display_sales_cols = df_sales.columns[2:]
display_sales_cols = [
        {'name': i, 'id': i, 'editable': False} for i in display_sales_cols 
    ]

display_som_cols = df_som.columns[2:]
display_som_cols = [
        {'name': i, 'id': i, 'editable': False} for i in display_som_cols 
    ]

#### Upper tile card values
df_sales_pred = pd.read_csv('./Data/sales_prediction.csv')
df_sales_pred['Date'] = pd.to_datetime(df_sales_pred['Date'])
df_sales_pred = df_sales_pred[df_sales_pred['Sales'].notna()]

max_date = df_sales_pred.Date.max()

#### Error plots
df_track = pd.DataFrame()
cols = ['Manufacturer', 'Channel', 'Date', 'SOM', 'SOM_hat', 'Sales', 'sales_pred']
for keys, df_slice in df_sales_pred.groupby(['Channel', 'Manufacturer']):
    df_slice = df_slice.sort_values(by='Date')[cols].tail(6)
    df_track = pd.concat([df_track, df_slice])

df_track2 = pd.DataFrame()
for keys, df_slice in df_track.groupby(['Manufacturer', 'Channel']):
    df_slice['sales MAPE'] = np.abs(df_slice['Sales'] - df_slice['sales_pred'])*100/np.abs(df_slice['Sales'])
    df_slice['SOM mae'] = np.abs(df_slice['SOM']-df_slice['SOM_hat'])
    df_track2 = pd.concat([df_track2, df_slice])
df_track2 = df_track2.sort_values(by=['Manufacturer', 'Channel', 'Date'])
df_track2 = df_track2[['Manufacturer', 'Channel', 'Date', 'SOM', 'SOM_hat', 'SOM mae', 
                     'Sales', 'sales_pred', 'sales MAPE']]
df_track2['Date'] = df_track2['Date'].apply(lambda x: x.strftime("%B %Y"))


# Importing Feature Importance Data
df_feat_imp = pd.read_csv('./Data/feature_importance_data.csv')

dash.register_page(__name__)

layout = dbc.Container([
    
    dbc.Row([
        dbc.ButtonGroup(
        [
            dbc.Button("Home", id= 'navigation-page',
                       style = {'width': '5in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/home'),
            dbc.Button("Simulation", id= 'first-page',
                       style = {'width': '5in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/Simulation',
                       className="ml-auto"),
            dbc.Button("Prediction Accuracy", className="ml-auto", id= 'second-page',
                       style = {'width': '5in', 'background-color': '#01529C', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'}
                       ),
        ], 
    )
    ], align = 'center', justify= 'center'),
    
    dbc.Row([
       dbc.Col([
           html.P(f"Data is available till {max_date.strftime('%b')}-{max_date.strftime('%Y')}", 
                  style= {'font-family': 'Verdana', 'font-size': '13px'}),
       ], width=3)
    ], justify="right", align="right"),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Channel', style= {'marginLeft': '0px', 'marginRight': '64px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-channel',
                options=[{"label": i, "value": i} for i in df_sales.Channel.unique()],
                         value='DTS', clearable = False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
        ], width=2),
        
        dbc.Col([
            html.Label('Manufacturer', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-manufacturer',
                options=[{"label": i, "value": i} for i in df_sales.Manufacturer.unique()],
                         value='PEPSICO', clearable= False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
        ], width=2),
        dbc.Col([
            html.Label('Parameter', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                          'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-parameter',
                        options=[{"label": i, "value": i} for i in ['SOM', 'SALES']],
                                 value='SOM', clearable= False, style={'color': 'black', 
                                                                      'marginLeft': '0px',
                                                                          'width': '210px', 'font-family': 'Verdana', 'font-size': '14px'})
        ])
        
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ], style= {'border-top': '2px solid grey', 'margin': '0px', 'margin-top': '5px'}),
    
    dbc.Row([
        dbc.Col([
            
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'Model Training Performance', style = {'font-weight': 'bold',
                                                                                       'color': '#1D4693', 'font-size': '100%',
                                                                                       'border': 'none', 'background-color': '#E8E8E8',
                                                                                      'width': '800px', 'font-family': 'Verdana'})
                ], horizontal=True),
                
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'WAPE (%)', style = {'font-weight': 'bold', 'color': '#F9F9F9', 
                                                                    'font-color': 'white', 'border': 'none', 
                                                                    'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(children= 'Last 1 Month', style = {'font-weight': 'bold', 'border': 'none',
                                                                         'width': '155px', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(children= 'Last 6 Months', style = {'font-weight': 'bold', 'width': '155px', 
                                                                          'border': 'none', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(children= 'Last 12 Months', style = {'font-weight': 'bold', 'border': 'none', 
                                                                           'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '14px'}),
                ], horizontal=True),
                
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'param-text', style= {'width': '120px', 'border': 'none', 'font-weight': 'bold', 
                                                                'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size': '14px'}),
                    dbc.ListGroupItem(id= 'last-month', style = {'background-color': '#01529C', 'width': '160px', 
                                                                 'font-weight': 'bold', 'color': 'white',
                                                                    'font-size': '14px', 'border-radius': '5px',
                                                                    'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9'}),
                    dbc.ListGroupItem(id= '6-months', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white', 
                                                               'font-weight': 'bold',
                                                               'font-size': '14px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center'}),
                    dbc.ListGroupItem(style = {'border': 'none', 'width': '5px', 'background-color': '#F9F9F9', 'font-family': 'Verdana'}),
                    dbc.ListGroupItem(id= '12-months', style = {'background-color': '#01529C', 'width': '160px',
                                                               'color': 'white', 
                                                               'font-weight': 'bold', 'width': '160px',
                                                               'font-size': '14px', 'border-radius': '5px',
                                                               'border-color': 'white', 'text-align': 'center', 'font-family': 'Verdana'}),
                    
                ], horizontal=True)
                
            ], style= {'border': 'none', 'background-color': '#F9F9F9'}
                
            ),
        ], width= 6),
        dbc.Col([
            dbc.Card([
                html.H4(id= 'error-graph-text', style= {'font-size': '15px', 'font-family': 'Verdana'}),
                dcc.Graph(id='error-graph', style= {'height': '40vh'})
            ], style= {'height': '40vh', 'marginLeft': '10px', 'border': 'none', 
                       'background-color': '#F9F9F9'}, body=True)
        ], width= 6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ], style= {'border-bottom': '2px solid grey', 'margin': '0px'}),
        
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'Model Performance Matrix', style= {'width': '100%', 'color': '#1D4693',
                                                                                      'background-color': '#E8E8E8', 'border': 'none', 
                                                                                     'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                ])
                
            ], style= {'border': 'none'}
                
            ),
        ], width= 12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='error-table',
                                style_header={ 'border': '1px solid white', 'whiteSpace':'normal', 'color': 'white',
                                              'font-weight': 'bold', 'backgroundColor': '#01529C', 'font-family': 'Verdana',
                                              'font-size':'10px'},
                                style_cell={ 'minWidth': 80, 'maxWidth': 120, 'border': '1px solid grey', 
                                            'font-family': 'Verdana', 'font-size':'10px'},
                                style_table={'overflowX': 'auto', 'height': '300px'}, 
                                # virtualization=True,
                                fixed_rows={'headers': True},
                                )
        ], width=12, style= {'font-size': '11px'}, align="center")
    ]),


], fluid = True, style = {'background-color': '#F9F9F9'})

@callback(
    Output("output-page", "children"), 
    [Input("page-tabs", "value")]
)
def display_value(value):
    return f"Selected value: {value}"

@callback(
    Output("param-text", 'children'),
    Output("last-month", 'children'),
    Output("6-months", 'children'),
    Output("12-months", 'children'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('chosen-parameter', 'value')
)
def card_values(mfg, cnl, parameter):
    global df_sales_pred
    
    df_temp = df_sales_pred[(df_sales_pred['Manufacturer']==mfg)&(df_sales_pred['Channel']==cnl)]
    
    df_last_month = df_temp[(df_temp['Date']==df_temp['Date'].max())]
    df_6_months = df_temp[(df_temp['Date'].isin(sorted(df_temp['Date'].unique())[-6:]))]
    df_12_months = df_temp[(df_temp['Date'].isin(sorted(df_temp['Date'].unique())[-12:]))]
    
    if parameter == 'SOM':
        text = 'MAE (%)'
        # last month
        last_month = mean_absolute_error(df_last_month['SOM'], df_last_month['SOM_hat'])
        last_month = '{:.2f}'.format(last_month)
        
        # last 6 months
        rolling_6_months = mean_absolute_error(df_6_months['SOM'], df_6_months['SOM_hat'])
        rolling_6_months = '{:.2f}'.format(rolling_6_months)
        
        # last 12 months
        rolling_12_months = mean_absolute_error(df_12_months['SOM'], df_12_months['SOM_hat'])
        rolling_12_months = '{:.2f}'.format(rolling_12_months)
    else:
        text = 'WAPE (%)'
        # last month
        last_month = WAPE(df_last_month['Sales'], df_last_month['sales_pred'])
        last_month = '{:.2f}'.format(last_month)
        
        # last 6 months
        rolling_6_months = WAPE(df_6_months['Sales'], df_6_months['sales_pred'])
        rolling_6_months = '{:.2f}'.format(rolling_6_months)
        
        # last 12 months
        rolling_12_months = WAPE(df_12_months['Sales'], df_12_months['sales_pred'])
        rolling_12_months = '{:.2f}'.format(rolling_12_months)
    
    return text, last_month, rolling_6_months, rolling_12_months
    

@callback(
    Output('error-graph-text', 'children'),
    Output('error-graph', 'figure'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('chosen-parameter', 'value')
)
def card_plot(mfg, cnl, parameter):
    global df_track2, df_feat_imp
    temp = df_track2[(df_track2['Manufacturer']==mfg)&
                                (df_track2['Channel']==cnl)]
    if parameter == 'SOM':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x= temp['Date'], y = temp['SOM mae'], 
                                     name= 'SOM Errors (%)', marker= dict(color= '#01529C')))
        fig.update_layout(yaxis_title = 'MAE (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                              margin= {'t': 0, 'b': 0, 'r': 0, 'l': 0})
        fig_text = f'{mfg} X {cnl} SOM Error (MAE %)'
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x= temp['Date'], y = temp['sales MAPE'], 
                                     name= 'Sales Errors (%)', marker= dict(color= '#01529C')))
        fig.update_layout(yaxis_title = 'WAPE (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                                margin= {'t': 0, 'b': 0, 'r': 0, 'l': 0})
        fig_text = f'{mfg} X {cnl} Sales Error (WAPE %)'
    
    return fig_text, fig


@callback(
    Output('error-table', 'data'),
    Output('error-table', 'columns'),
    Output('error-table', 'style_data_conditional'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('chosen-parameter', 'value')
)
def actualize_db(mfg, cnl, param):
    global display_sales_cols, display_som_cols, df_sales, df_som
    if param == 'SOM':
        df_slice = df_som[(df_som['Manufacturer']==mfg)&(df_som['Channel']==cnl)]
        display_cols = display_som_cols
        (styles, legend) = discrete_background_color_bins(df_slice)
    else:
        df_slice = df_sales[(df_sales['Manufacturer']==mfg)&(df_sales['Channel']==cnl)]
        display_cols = display_sales_cols
        (styles, legend) = discrete_background_color_bins(df_slice)
    
    return df_slice.to_dict('records'), display_cols, styles