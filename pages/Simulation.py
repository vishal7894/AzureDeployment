#!/usr/bin/env python
# coding: utf-8


import datetime
import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm
from statsmodels.tools.tools import add_constant

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm
from statsmodels.tools.tools import add_constant

# Removing redundant features based on their statistical significance
def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features

# MAPE metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))*100

def WAPE(y_true, y_pred):
    return (y_true - y_pred).abs().sum()*100 / y_true.abs().sum()

# sort column order
sortList = ['PEPSICO PPU', 'PEPSICO TDP', 'COMESTIBLES PPU', 'COMESTIBLES TDP', 'RAMO PPU', 'RAMO TDP',
 'YUPI PPU', 'YUPI TDP', 'OTROS PPU', 'OTROS TDP']
def sort_fun(x):
    for i, thing in enumerate(sortList):
        if x.startswith(thing):
            return (i, x[len(thing):])

# Engineered features
regression_features = ['COMESTIBLES PPU', 'OTROS PPU', 'PEPSICO PPU',
                       'RAMO PPU', 'YUPI PPU', 'COMESTIBLES TDP', 'OTROS TDP',
                       'PEPSICO TDP', 'RAMO TDP', 'YUPI TDP',
                       'COMESTIBLES PPU Ratio', 'OTROS PPU Ratio',
                       'PEPSICO PPU Ratio', 'RAMO PPU Ratio',
                       'YUPI PPU Ratio', 'COMESTIBLES TDP Ratio', 'OTROS TDP Ratio',
                       'PEPSICO TDP Ratio', 'RAMO TDP Ratio', 'YUPI TDP Ratio']

def generate_comp_features_1(data):
    df_comp = pd.DataFrame()
    for keys, df_slice in data.groupby(['Date', 'Channel']):
        df_t = df_slice.pivot( columns = 'Manufacturer', values = ['PPU'])
        df_t.columns =  [str(j + " " + i) for i, j in df_t.columns]
        df_t = df_t.fillna(0).sum()
        new_cols = df_t.index
        new_vals = df_t.values
        new_vals = df_t.values
        df_slice[new_cols] = new_vals

        df_t = df_slice.pivot( columns = 'Manufacturer', values = ['TDP'])
        df_t.columns =  [str(j + " " + i) for i, j in df_t.columns]
        df_t = df_t.fillna(0).sum()
        new_cols = df_t.index
        new_vals = df_t.values
        new_vals = df_t.values
        df_slice[new_cols] = new_vals
        df_comp = df_comp.append(df_slice)
    return df_comp

def generate_comp_features_2(data):
    data['Month'] = data['Date'].dt.month
    avg_cols = ['COMESTIBLES PPU', 'OTROS PPU', 'PEPSICO PPU', 'RAMO PPU', 'YUPI PPU']
    tdp_cols = ['COMESTIBLES TDP', 'OTROS TDP', 'PEPSICO TDP', 'RAMO TDP', 'YUPI TDP']

    for col in avg_cols:
        data[f'{col} Ratio'] = data[col].div(data.PPU)

    for col in tdp_cols:
        data[f'{col} Ratio'] = data[col].div(data.TDP)
    
    return data


from sktime.forecasting.exp_smoothing import ExponentialSmoothing

def kpi_prediction_exp_smoothing(future_days, kpi, data):
    data = data[['Manufacturer', 'Channel', 'Date', kpi]]
    data['Date'] = pd.to_datetime(data['Date'])
    
    df_pred = pd.DataFrame()
    for keys, df_grp in data.groupby(['Manufacturer', 'Channel']):
        mfg, cnl = keys
        df_grp = df_grp.sort_values(by='Date')
        train_data = df_grp.copy()
        train_data = train_data[['Date', kpi]].set_index('Date', drop=True)
        train_data[kpi] = train_data[kpi].astype('float64')
        train_data = train_data.to_period(freq="M")
        model = ExponentialSmoothing(trend='mul', seasonal='multiplicative', sp=12)
        model.fit(train_data)
        
        future_pred = pd.DataFrame(model.predict(fh=np.arange(1, future_days+1)))
        future_pred.columns = [f'{kpi}_hat']
        future_pred = future_pred.to_timestamp().reset_index().rename(columns= {'index': 'Date'})
        future_pred[['Manufacturer', 'Channel']] = mfg, cnl
        df_grp = df_grp.merge(future_pred, on= ['Date', 'Manufacturer', 'Channel'], how= 'outer')

        df_pred = pd.concat([df_pred, df_grp])
    return df_pred

def plot_dist_charts(pred_data, mfg):
    pred_data = pred_data[['Date', 'Manufacturer', 'Channel', 'Sales', 'SOM', 'Predicted Sales', "Predicted SOM"]]
    pred_data['Date'] = pd.to_datetime(pred_data['Date'])
    pred_data['month'] = pred_data['Date'].dt.month
    pred_data['year'] = pred_data['Date'].dt.year

    pred_data = pred_data[pred_data['year']!=2019]
    pred_data_1 = pred_data.groupby(['year', 'Channel', 'Manufacturer']).sum().reset_index()[['year', 
                                                                                 'Channel', 
                                                                                 'Manufacturer', 
                                                                                 'Sales','Predicted Sales']]

    pred_data_1['sales_agg'] = pred_data_1.groupby('year')['Sales'].transform('sum')
    pred_data_1['SOG'] = pred_data_1['Sales']*100/pred_data_1['sales_agg']

    pred_data_1['pred_sales_agg'] = pred_data_1.groupby('year')['Predicted Sales'].transform('sum')
    pred_data_1['pred_SOG'] = pred_data_1['Predicted Sales']*100/pred_data_1['pred_sales_agg']

    pred_data_1 = round(pred_data_1,2)

    colors = {
                "A": '#0096D6',
                "B": '#00984A', 
                "C": '#EB7B30',
                "D": '#005CB4',
                "E": '#C9002B', 
                "F": '#00984A'
                }

    year_chart_df = pred_data_1.pivot(index = ['year', 'Manufacturer'], 
                                      columns= 'Channel', 
                                      values = ['SOG', 'pred_SOG']).reset_index()
    year = year_chart_df.year.max()

    # filter for manufacturer
    year_chart_df = year_chart_df[year_chart_df[('Manufacturer', '')] == mfg]
    last_sales_year = pred_data_1[pred_data_1['Sales']==0]['year'].min() - 1

    act_year_df = year_chart_df[year_chart_df[('year','')]<=last_sales_year]
    years = [str(i) for i in act_year_df.year.unique()]
    years[-1] = '2022 YTD'

    plot = go.Figure(data=[go.Bar(
                            name = 'DTS',
                            x = years,
                            y = act_year_df[('SOG','DTS')],
                            text =  act_year_df[('SOG','DTS')],
                            marker_color=colors['D']
                            ),
                            go.Bar(
                            name = 'SUPERCADENAS',
                            x = years,
                            y = act_year_df[('SOG','SUPERCADENAS')],
                            text = act_year_df[('SOG','SUPERCADENAS')],
                            marker_color=colors['E']
                            ),
                            go.Bar(
                            name = 'SUPERETES',
                            x = years,
                            y = act_year_df[('SOG','SUPERETES')], 
                            text = act_year_df[('SOG','SUPERETES')],
                            marker_color=colors['F']
                            )])
    
    est_year_df = year_chart_df[year_chart_df[('year','')]>=last_sales_year]
    
    estimated_df_years = list(est_year_df[('year','')].unique())
    # estimated_df_years = [f'{i} estimated' for i in estimated_df_years]
    act_year_df[('SOG', 'sum')] = act_year_df[('SOG',
                                           'SUPERCADENAS')] + act_year_df[('SOG',
                                                                           'DTS')] + act_year_df[('SOG',
                                                                                                  'SUPERETES')]
    est_year_df[('pred_SOG', 'sum')] = est_year_df[('pred_SOG',
                                            'SUPERCADENAS')] + est_year_df[('pred_SOG',
                                                                            'DTS')] + est_year_df[('pred_SOG',
                                                                                                    'SUPERETES')]
    if len(act_year_df.year)>0:
            plot.add_annotation(x = len(act_year_df.year)/2, text = 'Actual', 
                            y = act_year_df[('SOG','sum')].max()+act_year_df[('SOG','sum')].max()*0.1, showarrow = False)

    if estimated_df_years:
        plot.add_traces([go.Bar(
                            name = 'DTS',
                            x = estimated_df_years,
                            y = est_year_df[('pred_SOG', 'DTS')],
                            text =  est_year_df[('pred_SOG', 'DTS')],
                            marker_color=colors['D'], showlegend=False
                            ), 
                            go.Bar(
                            name = 'SUPERCADENAS',
                            x = estimated_df_years,
                            y = est_year_df[('pred_SOG', 'SUPERCADENAS')],
                            text = est_year_df[('pred_SOG', 'SUPERCADENAS')],
                            marker_color=colors['E'], showlegend=False
                            ), 
                            go.Bar(
                            name = 'SUPERETES',
                            x = estimated_df_years,
                            y = est_year_df[('pred_SOG', 'SUPERETES')], 
                            text = est_year_df[('pred_SOG', 'SUPERETES')],
                            marker_color=colors['F'], showlegend=False)])

        plot.add_vline(x = 2.5, line_dash = 'dot')
        plot.add_annotation(x = len(act_year_df.year)-0.5+ (len(act_year_df.year) + len(est_year_df.year) - len(act_year_df.year))/2, text = 'Estimated', 
                        y = act_year_df[('SOG','sum')].max()+act_year_df[('SOG','sum')].max()*0.1, showarrow = False)

    plot.update_layout(barmode='stack', title = f'{mfg} YoY SOG(Source of Growth) SOM', 
                           paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',)
    return plot

import calendar
def plot_monthly_dist_chart(pred_data, mfg, year):
    pred_data_2 = pred_data.groupby(['Date', 'Channel', 'Manufacturer']).sum().reset_index()[['Date', 
                                                                                 'Channel', 
                                                                                 'Manufacturer', 
                                                                                 'Sales','Predicted Sales']]
    
    colors = {
                "A": '#0096D6',
                "B": '#00984A', 
                "C": '#EB7B30',
                "D": '#005CB4',
                "E": '#C9002B', 
                "F": '#00984A'
                }

    pred_data_2['sales_agg'] = pred_data_2.groupby('Date')['Sales'].transform('sum')
    pred_data_2['SOG'] = pred_data_2['Sales']*100/pred_data_2['sales_agg']

    pred_data_2['pred_sales_agg'] = pred_data_2.groupby('Date')['Predicted Sales'].transform('sum')
    pred_data_2['pred_SOG'] = pred_data_2['Predicted Sales']*100/pred_data_2['pred_sales_agg']

    last_month_sales = pred_data_2[pred_data_2['Sales']!=0].Date.max()

    monthly_chart_df = pred_data_2.pivot(index=['Date','Manufacturer'], columns = 'Channel',
                                   values = ['SOG', 'pred_SOG']).reset_index()

    monthly_chart_df = round(monthly_chart_df, 2)
    monthly_chart_df['month'] = monthly_chart_df['Date'].dt.month
    monthly_chart_df['year'] = monthly_chart_df['Date'].dt.year

    monthly_chart_df['month_name'] = monthly_chart_df['month'].apply(lambda x:calendar.month_name[x])
    monthly_chart_df.drop("month", axis=1, inplace=True)
    monthly_chart_df.rename(columns= {"month_name":"month"}, inplace=True)

    monthly_chart_df = monthly_chart_df[(monthly_chart_df[('Manufacturer','')] == mfg) &
                                        (monthly_chart_df[('year','')] == year)
                                        ]

    # filter for manufacturer
    actual_df = monthly_chart_df[monthly_chart_df[('Date','')] <= last_month_sales]
    

    plot = go.Figure(data=[go.Bar(
                            name = 'DTS',
                            x = actual_df.month,
                            y = actual_df[('SOG','DTS')],
                            text =  actual_df[('SOG','DTS')],
                            marker_color=colors['D']
                            ), 
                            go.Bar(
                            name = 'SUPERCADENAS',
                            x = actual_df.month,
                            y = actual_df[('SOG','SUPERCADENAS')],
                            text = actual_df[('SOG','SUPERCADENAS')],
                            marker_color=colors['E']
                            ), 
                            go.Bar(
                            name = 'SUPERETES',
                            x = actual_df.month,
                            y = actual_df[('SOG','SUPERETES')], 
                            text = actual_df[('SOG','SUPERETES')],
                            marker_color=colors['F']
                           )])

    actual_df[('SOG',
              'sum')] = actual_df[('SOG', 
                                    'DTS')] + actual_df[('SOG', 
                                                        'SUPERCADENAS')] + actual_df[('SOG',
                                                                                        'SUPERETES')]
    if len(actual_df.month)>0:
        plot.add_annotation(x = len(actual_df.month)/2, text = 'Actual', 
                        y = actual_df[('SOG','sum')].max()+actual_df[('SOG','sum')].max()*0.1, showarrow = False)
    estimated_df = monthly_chart_df[monthly_chart_df[('Date','')] > last_month_sales]
    estimated_df_months = list(estimated_df[('month','')].unique())
    # estimated_df_months = [f'{i} estimated' for i in estimated_df_months]
    estimated_df[('pred_SOG',
              'sum')] = estimated_df[('pred_SOG', 
                                    'DTS')] + estimated_df[('pred_SOG', 
                                                        'SUPERCADENAS')] + estimated_df[('pred_SOG',
                                                                                        'SUPERETES')]

    if estimated_df_months:
        plot.add_traces([go.Bar(
                                name = 'DTS',
                                x = estimated_df_months,
                                y = estimated_df[('pred_SOG', 'DTS')],
                                text = estimated_df[('pred_SOG', 'DTS')],
                                marker_color=colors['D'], showlegend=False
                                ), 
                                go.Bar(
                                name = 'SUPERCADENAS',
                                x = estimated_df_months,
                                y = estimated_df[('pred_SOG', 'SUPERCADENAS')],
                                text = estimated_df[('pred_SOG', 'SUPERCADENAS')],
                                marker_color=colors['E'], showlegend=False
                                ), 
                                go.Bar(
                                name = 'SUPERETES',
                                x = estimated_df_months,
                                y = estimated_df[('pred_SOG', 'SUPERETES')], 
                                text = estimated_df[('pred_SOG', 'SUPERETES')],
                                marker_color=colors['F'], showlegend=False
                               )])

        plot.add_vline(x = len(actual_df.month)-0.5, line_dash = 'dot')
        plot.add_annotation(x = len(actual_df.month)-0.5+ (12 - len(actual_df.month))/2, text = 'Estimated', 
                        y = actual_df[('SOG','sum')].max()+(actual_df[('SOG','sum')].max())/10, showarrow = False)
        # plot.add_annotation(x = (len(actual_df.month))/2, text = 'Actual', 
        #                 y = estimated_df[('pred_SOG','sum')].max()+5, showarrow = False)
    plot.update_layout(barmode='stack', title = f'{mfg} MoM SOG(Source of Growth) SOM for {year}', 
                      paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9')
    return plot

def diff_dashtable(data, data_previous, row_id_name=None):
    """Generate a diff of Dash DataTable data.

    Modified from: https://community.plotly.com/t/detecting-changed-cell-in-editable-datatable/26219/2

    Parameters
    ----------
    data: DataTable property (https://dash.plot.ly/datatable/reference)
        The contents of the table (list of dicts)
    data_previous: DataTable property
        The previous state of `data` (list of dicts).

    Returns
    -------
    A list of dictionaries in form of [{row_index:, column_id:, current_value:,
        previous_value:}]
    """
    df, df_previous = pd.DataFrame(data=data), pd.DataFrame(data_previous)

    if row_id_name is not None:
        # If using something other than the index for row id's, set it here
        for _df in [df, df_previous]:

            # Why do this?  Guess just to be sure?
            assert row_id_name in _df.columns

            _df = _df.set_index(row_id_name)
    else:
        row_id_name = "row_index"

    # Mask of elements that have changed, as a dataframe.  Each element indicates True if df!=df_prev
    df_mask = ~((df == df_previous) | ((df != df) & (df_previous != df_previous)))

    # ...and keep only rows that include a changed value
    df_mask = df_mask.loc[df_mask.any(axis=1)]

    changes = []

    # This feels like a place I could speed this up if needed
    for idx, row in df_mask.iterrows():
        row_id = row.name

        # Act only on columns that had a change
        row = row[row.eq(True)]

        for change in row.iteritems():

            changes.append(
                {
                    row_id_name: row_id,
                    "column_id": change[0],
                }
            )

    return changes

# # With forecasted regressors

# Importing Scantrack data
df = pd.read_csv('./Data/scantrack_base_monthly_v2.csv')
del df['date']
df.columns = ['Manufacturer', 'PPU', 'TDP', 'Sales', 'Units', 'Date', 'Channel', 'SOM']
df['Date'] = pd.to_datetime(df['Date'])
max_date = df['Date'].max()
print(df.shape)


future_days = 18
df_ppu= kpi_prediction_exp_smoothing(future_days, 'PPU', df)
df_tdp= kpi_prediction_exp_smoothing(future_days, 'TDP', df)
df_tdp['TDP_hat'] = np.ceil(df_tdp['TDP_hat'])


df_pred_whole = df_ppu.merge(df_tdp, on = ['Manufacturer', 'Channel', 'Date'], how= 'left')


df_reg_pred = df_pred_whole[df_pred_whole['Date']>max_date]
df_reg_pred = df_reg_pred.drop(columns= ['PPU', 'TDP']).rename(columns= {'PPU_hat': 'PPU', 
                                                                               'TDP_hat': 'TDP'})

# ## Forecasting sales on Prediction data

# Importing Scantrack data
df = pd.read_csv('./Data/scantrack_base_monthly_v2.csv')
del df['date']
df.columns = ['Manufacturer', 'PPU', 'TDP', 'Sales', 'Units', 'Date', 'Channel', 'SOM']
df['Date'] = pd.to_datetime(df['Date'])
print(df.shape)

max_date = df.Date.max()

df = pd.concat([df, df_reg_pred])
print(df.shape)


# Adding competitor price and TDP as features
df_comp = pd.DataFrame()
for keys, df_slice in df.groupby(['Date', 'Channel']):
    df_t = df_slice.pivot( columns = 'Manufacturer', values = ['PPU'])
    df_t.columns =  [str(j + " " + i) for i, j in df_t.columns]
    df_t = df_t.fillna(0).sum()
    new_cols = df_t.index
    new_vals = df_t.values
    new_vals = df_t.values
    df_slice[new_cols] = new_vals
    
    df_t = df_slice.pivot( columns = 'Manufacturer', values = ['TDP'])
    df_t.columns =  [str(j + " " + i) for i, j in df_t.columns]
    df_t = df_t.fillna(0).sum()
    new_cols = df_t.index
    new_vals = df_t.values
    new_vals = df_t.values
    df_slice[new_cols] = new_vals
    df_comp = df_comp.append(df_slice)

# Feature engineering
df_comp['Date'] = pd.to_datetime(df_comp['Date'])
df_comp['Month'] = df_comp['Date'].dt.month
avg_cols = ['COMESTIBLES PPU', 'OTROS PPU', 'PEPSICO PPU', 'RAMO PPU', 'YUPI PPU']
tdp_cols = ['COMESTIBLES TDP', 'OTROS TDP', 'PEPSICO TDP', 'RAMO TDP', 'YUPI TDP']

for col in avg_cols:
    df_comp[f'{col} Ratio'] = df_comp[col].div(df_comp.PPU)

for col in tdp_cols:
    df_comp[f'{col} Ratio'] = df_comp[col].div(df_comp.TDP)

for col in avg_cols:
    df_comp[f'{col} gap'] = np.abs(df_comp['PPU'] - df_comp[col])

for col in tdp_cols:
    df_comp[f'{col} gap'] = np.abs(df_comp['TDP'] - df_comp[col])


# Fitting Linear Regression model
model_feat_dict = {}
model_dict = {}
df_sales_pred = pd.DataFrame()
future_months = 18
for keys, df_slice in df_comp.groupby(['Manufacturer', 'Channel']):
    mfr, cnl = keys
    df_slice = df_slice.sort_values(by='Date')
    oh_cols = list(pd.get_dummies(df_slice['Month']).columns.values)[1:]
    encoded_features = pd.get_dummies(df_slice['Month'])
    df_slice2 = pd.concat([df_slice, encoded_features],axis=1)
    df_slice2 = df_slice2.sort_values(by='Date')
    df_slice3 = df_slice2.iloc[:-1*future_months]
    X, y = df_slice3[regression_features + oh_cols], df_slice3[['Sales']]
    X[regression_features] = X[regression_features].apply(np.log10)
    
    # Feature selection
    selected_features = backward_elimination(X, y)
    
    X = X[selected_features]
    
    model = LinearRegression()
    model.fit(X, y)
    model_dict[mfr+cnl] = model
    model_feat_dict[mfr+'_'+cnl] = selected_features

    df_slice2[regression_features] = df_slice2[regression_features].apply(np.log10)
    df_slice['sales_pred'] = model.predict(df_slice2[selected_features])
    df_sales_pred = pd.concat([df_sales_pred, df_slice])
    

# Calculate Predicted SOM
df_sales_pred['total_sales_hat'] = df_sales_pred.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
df_sales_pred['SOM_hat'] = df_sales_pred['sales_pred']*100/df_sales_pred['total_sales_hat']
del df_sales_pred['total_sales_hat']

df_all = df_sales_pred.groupby(['Manufacturer', 'Date'])[['Sales', 'sales_pred']].sum().reset_index()
df_all['Channel'] = 'ALL'
df_all['total_sales'] = df_all.groupby(['Date', 'Channel'])['Sales'].transform('sum')
df_all['SOM'] = df_all['Sales']*100/df_all['total_sales']
del df_all['total_sales']
df_all['total_sales_hat'] = df_all.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
df_all['SOM_hat'] = df_all['sales_pred']*100/df_all['total_sales_hat']
del df_all['total_sales_hat']


# Feature Importance Table
df_feat_imp = pd.DataFrame()

for keys, df_slice in df_comp.groupby(['Manufacturer', 'Channel']):
    mfg, cnl = keys
    df_slice = df_slice[df_slice['Sales'].notna()]
    df_one = df_slice.sort_values(by='Date')
    oh_cols = list(pd.get_dummies(df_one['Month']).columns.values)[1:]
    df_slice2 = df_one.join(pd.get_dummies(df_one['Month']))

    X, y = df_slice2[regression_features + oh_cols], df_slice2[['Sales']]
    X[regression_features] = X[regression_features].apply(np.log10)

    model = model_dict[mfg+cnl]
    selected_features = model_feat_dict[mfg+'_'+cnl]
    weights = model.coef_[0]


    X[selected_features] = np.multiply(X[selected_features], weights)
    X['sum'] = X[selected_features].abs().sum(axis = 1)
    for feature in selected_features:
        X[feature] = (X[feature])/X['sum']

    sel_oh_cols = list(set(selected_features).difference(set(regression_features)))
    sel_reg_features = list(set(regression_features).intersection(set(selected_features)))
    X['Month'] = X[sel_oh_cols].sum(axis = 1)
    X = X.drop(columns= sel_oh_cols)

    X_array = np.array(X[sel_reg_features + ['Month']])

    feature_names = sel_reg_features + ['Month']

    rf_resultX = pd.DataFrame(X_array, columns = feature_names)

    vals = rf_resultX.values.mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['Feature','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    shap_importance = shap_importance[(shap_importance['feature_importance_vals']>0.001)|
                                  (shap_importance['feature_importance_vals']<-0.001)]
    shap_importance = shap_importance.sort_values(by = 'feature_importance_vals',  ascending= True)

    shap_importance["Color"] = np.where(shap_importance["feature_importance_vals"]<0, '#C9002B', '#005CB4')
    shap_importance[['Manufacturer', 'Channel']] = mfg, cnl
    
    df_feat_imp = pd.concat([df_feat_imp, shap_importance])

df_feat_imp.to_csv('./Data/feature_importance_data.csv', index=False)

# ##  5. Dash 

# Importing Scantrack data
df = pd.read_csv('./Data/scantrack_base_monthly_v2.csv')
del df['date']
df.columns = ['Manufacturer', 'PPU', 'TDP', 'Sales', 'Units', 'Date', 'Channel', 'SOM']
df['Date'] = pd.to_datetime(df['Date'])
print(df.shape)

df_dist = df.copy()
df_dist['Year'] = df_dist.Date.dt.year
year = df_dist.Year.unique()
year = [str(i) for i in year]

df_  = df_dist[df_dist['Year']==2022]
dist_plot = px.sunburst(df_.assign(hole=" "), path=['hole','Channel',
                                             'Manufacturer'], values='Sales',
                  labels='Sales', 
                  color='Sales', hover_data=['Sales'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df_['Sales'], 
                                                       weights=df_['Sales']
                                                      ))

dist_plot.update_layout(title_text = "Sales Distribution 2022", paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9')


max_date = df.Date.max()

df = pd.concat([df, df_reg_pred])
print(df.shape)

df_comp = generate_comp_features_1(df)


df_comp = df_comp.round(decimals = 2)


df_sales_pred = df_sales_pred.round(decimals = 2)
df_sales_pred.to_csv('./Data/sales_prediction.csv', index=False)



display_cols = [
         {'name': 'Manufacturer', 'id': 'Manufacturer', 'editable': False},
         {'name': 'Channel', 'id': 'Channel', 'editable': False},
         {'name': 'Date', 'id': 'Date', 'editable': False},
         {'name': 'Month', 'id': 'Month', 'editable': False},
         {'name': 'Sales', 'id': 'Sales', 'editable': False},
         {'name': 'SOM', 'id': 'SOM', 'editable': False}
            ]


## Dash

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_table
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from dateutil.relativedelta import relativedelta
from dash import callback

dash.register_page(__name__, path='/')

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

layout = dbc.Container([
    
    dbc.Row([
        dbc.ButtonGroup(
        [
            dbc.Button("Home", id= 'navigation-page',
                       style = {'width': '5in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/home'),
            dbc.Button("Simulation", id= 'first-page',
                       style = {'width': '5in', 'background-color': '#01529C', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       className="ml-auto"),
            dbc.Button("Prediction Accuracy", className="ml-auto", id= 'second-page',
                       style = {'width': '5in', 'background-color': 'rgba(1, 82, 156, 0.7)', 'color': 'white', 
                               'font-weight': 'bold', 'border-color': 'white', 'font-family': 'Verdana', 'font-size': '15px'},
                       href='/ErrorMetrics'),
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
                options=[{"label": i, "value": i} for i in ['DTS', 'SUPERCADENAS', 'SUPERETES', 'ALL']],
                         value='DTS', clearable = False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
        ], width=2),
        
        dbc.Col([
            html.Label('Manufacturer', style= {'marginLeft': '0px', 'marginRight': '50px', 'font-family': 'Verdana', 'font-size': '14px',
                                                'font-weight': 'bold'}),
            dcc.Dropdown(id='chosen-manufacturer',
                options=[{"label": i, "value": i} for i in df_comp.Manufacturer.unique()],
                         value='PEPSICO', clearable= False, style={'color': 'black', 'font-family': 'Verdana', 'font-size': '14px'}),
        ], width=2),
        
        dbc.Col([
            dbc.Card(
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'som-ytd', style = {'font-weight': 'bold', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '15px'}),
                    dbc.ListGroupItem(id= 'som-ytd-text', style = {'background-color': '#01529C',
                                                                    'color': 'white', 
                                                                    'font-weight': 'bold',
                                                                    'font-size': '11px', 'font-family': 'Verdana'}),
                ], flush = True),
                style= {'marginLeft': '120px', 'marginRight': '0px', 'text-align': 'center',
                       'marginTop': '0px', 'border-color': '#F9F9F9'}
            ),
        ], width = 3),
        
        dbc.Col([
            dbc.Card(
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'som-next', style = {'font-weight': 'bold', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '15px'}),
                    dbc.ListGroupItem(id= 'som-next-text', style = {'background-color': '#01529C',
                                                                    'color': 'white', 
                                                                    'font-weight': 'bold',
                                                                    'font-size': '11px', 'font-family': 'Verdana'}),
                ], flush = True),
                style= {'marginLeft': '0px', 'marginRight': '0px', 'text-align': 'center',
                       'marginTop': '0px', 'border-color': '#F9F9F9'}
            ),
        ], width = 2),
        dbc.Col([
            dbc.Card(
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'som-next-year', style = {'font-weight': 'bold', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '15px'}),
                    dbc.ListGroupItem(id= 'som-next-year-text', style = {'background-color': '#01529C',
                                                                         'color': 'white', 
                                                                         'font-weight': 'bold', 
                                                                         'font-size': '11px', 'font-family': 'Verdana'}),
                ], flush = True), 
                    style= {'marginLeft': '0px', 'text-align': 'center', 'border-color': '#F9F9F9'})
        ], width = 2)
        
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Img(id= 'channel-logo', width="180", height="150")
        ], width=2),
        dbc.Col([
            html.Img(id= 'manufacturer-logo', width="180", height="150", 
                     style= {'margin': 'auto', 'text-align': 'center'})
        ], width=2),
        dbc.Col([
            dbc.Card(
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'sales-ytd', style = {'font-weight': 'bold', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '15px'}),
                    dbc.ListGroupItem(id= 'sales-ytd-text', style = {'background-color': '#01529C',
                                                                     'color': 'white', 
                                                                     'font-weight': 'bold', 
                                                                     'font-size': '11px', 'font-family': 'Verdana'}),
                ], flush = True),
                style= {'marginLeft': '120px', 'marginRight': '0px', 'text-align': 'center',
                       'marginTop': '0px', 'border-color': '#F9F9F9'}
            ),
        ], width = 3),
        dbc.Col([
            dbc.Card(
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'sales-next', style = {'font-weight': 'bold', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '15px'}),
                    dbc.ListGroupItem(id= 'sales-next-text', style = {'background-color': '#01529C',
                                                                     'color': 'white', 
                                                                     'font-weight': 'bold', 
                                                                     'font-size': '11px', 'font-family': 'Verdana'}),
                ], flush = True),
                style= {'marginLeft': '0px', 'marginRight': '0px', 'text-align': 'center',
                       'marginTop': '0px', 'border-color': '#F9F9F9'}
            ),
        ], width = 2),
        dbc.Col([
            dbc.Card(
                dbc.ListGroup([
                    dbc.ListGroupItem(id= 'sales-next-year', style = {'font-weight': 'bold', 'background-color': '#F9F9F9', 'font-family': 'Verdana','font-size': '15px'}),
                    dbc.ListGroupItem(id= 'sales-next-year-text', style = {'background-color': '#01529C',
                                                                          'color': 'white', 
                                                                          'font-weight': 'bold', 
                                                                          'font-size': '11px', 'font-family': 'Verdana'}),
                ], flush = True), 
                    style= {'marginLeft': '0px', 'text-align': 'center', 'border-color': '#F9F9F9'})
        ], width = 2)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='som-graph')
        ], width=6, style = {'border-right': '2px solid grey', 'margin-top': '5px', 'margin-bottom': '10px'}),
        dbc.Col([
            dcc.Graph(id='sales-graph')
        ], width=6)
    ], style = {'border-top': '2px solid grey', 'border-bottom': '2px solid grey', 'margin': '0px'}), 
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.ListGroup([
                    dbc.ListGroupItem(children= 'SOM Simulation Table', style= {'width': '100%', 'color': '#1D4693',
                                                                                      'background-color': '#E8E8E8', 'border': 'none', 
                                                                                     'font-weight': 'bold', 'font-size': '100%', 'font-family': 'Verdana'}),
                ])
                
            ], style= {'border': 'none'}
                
            ),
        ], width= 12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Button(
            "Export", id="export-button", n_clicks=0, style= {'width': '50px', 'height': '23px', 
                                                            'border-radius': '0px', 'font-size': '11px', 
                                                            'text-align': 'center', 'color': 'buttontext', 
                                                            'background-color': 'buttonface', 'font-family': 'Verdana',
                                                            'padding': '1px 6px', 'border-width': '2px',
                                                            'border-style': 'outset', 'border-color': 'buttonborder',
                                                            'border-image': 'initial', 'display': 'inline-block', 
                                                            'margin-bottom': '2px'}
        ),
            dcc.Download(id="download-dataframe-csv"),
            
            dbc.Button(
            "Reset", id="reset-button", n_clicks=0, style= {'width': '50px', 'height': '23px', 
                                                            'border-radius': '0px', 'font-size': '11px', 
                                                            'text-align': 'center', 'color': 'buttontext', 
                                                            'background-color': 'buttonface', 'font-family': 'Verdana',
                                                            'padding': '1px 6px', 'border-width': '2px',
                                                            'border-style': 'outset', 'border-color': 'buttonborder',
                                                            'border-image': 'initial', 'display': 'inline-block', 
                                                            'margin-bottom': '2px', 'margin-top': '1px'}
        )
        ], width= 2),
        # dbc.Col([
        #     html.I(className="fa fa-long-arrow-left", style= {'font-size': '24px', 'color': 'white',
        #                                                   'background-color': '#F07836', 
        #                                                   'height': '23px'})
            
        #     ], width= 1.5, style= {'margin-left': '70%'}),
        # dbc.Col([
        #     html.P(
        #     "Simulation Columns", style= {'width': '100%', 'height': '23px', 
        #                                 'border-radius': '0px', 'font-size': '12px', 'font-family': 'Verdana',
        #                                 'text-align': 'center', 'color': 'white', 
        #                                 'background-color': '#F07836', 'font-weight': 'bold',
        #                                 'border': 'none','display': 'inline-block', 
        #                                 'margin-bottom': '2px', 'display': 'block', 'margin-left': '10px'}
        # )
            
        # ], width= 3.5, style= {'background-color': '#F07836', 'height': '23px'})
        dbc.Col([
            dbc.Card(style = {'width': '12px', 'height': '12px', 'margin-top': '8px',
                                'background-color': '#01529C', 'border': 'none', 'border-radius': '0px'}),
        ], width= 1.5, style= {'margin-left': '20%'}),
        dbc.Col([
            dbc.Card("Fixed Columns", style = {'width': '120px', 'height': '12px', 'margin-top': '3px', 'background-color': '#F9F9F9',
                                'color': 'black', 'border': 'none', 'font-family': 'Verdana', 'font-size': '13px'}),
        ], width= 2, style= {'width': '100px'}),
        dbc.Col([
            dbc.Card(style = {'width': '12px', 'height': '12px', 'margin-top': '8px',
                                'background-color': '#F07836', 'border': 'none', 'border-radius': '0px'}),
        ], width= 1.5),
        dbc.Col([
            dbc.Card("Simulation Columns", style = {'height': '12px', 'margin-top': '3px', 'background-color': '#F9F9F9',
                                'color': 'black', 'border': 'none', 'font-family': 'Verdana', 'font-size': '13px'}),
        ], width= 2),
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='table-editing-simple',
                                style_header={ 'border': '1px solid white', 'whiteSpace':'normal', 'color': 'white',
                                              'font-weight': 'bold', 'backgroundColor': '#01529C', 'font-family': 'Verdana',
                                              'font-size':'10px'},
                                style_cell={ 'border': '1px solid grey', 'minWidth': 85, 'maxWidth': 120, 
                                            'background-color': '#F9F9F9', 'font-family': 'Verdana', 'font-size':'10px'},
                                style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'}, 
                                virtualization=True,
                                fixed_rows={'headers': True},
                                style_header_conditional=[{
                                    'if': {'column_editable': True},
                                    'backgroundColor': '#F17836',
                                    'color': 'white'
    }]
                                )
        ], width=12, style= {'font-size': '11px'}, align="center")
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Br()
        ], width=12)
    ], style= {'border-top': '1px solid grey', 'margin': '0px'}),
    
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                options=[
                    {"label": "Monthly", "value": "Monthly"},
                    {"label": "Yearly", "value": "Yearly"},
                ], value= 'Monthly',
                id="time-selector",
                labelClassName="date-group-labels",
                labelCheckedClassName="date-group-labels-checked",
                className="date-group-items",
                inline=True
        ),
            dcc.Graph(id= 'dist-plot')
        ], width=6, style = {'border-right': '2px solid grey', 'margin-top': '5px', 'margin-bottom': '10px'}),
        dbc.Col([
            dcc.Graph(id='sensitivity-graph'),
            html.P("Above chart will provide users with the primary variables which will affect Sales & SOM in the data table above.", 
                   style= {'height': '20px', 'font-family': 'Verdana', 'font-size': '13px', 'margin-bottom': '5px'})
        ], width=6)
    ], style= {'border-top': '2px solid grey', 'margin': '0px', 'border-bottom': '2px solid grey'}),
    
    # dbc.Row([
    #     dbc.Col([
    #         dcc.Graph(id='monthly-dist-plot')
    #     ], width=6)
    # ], justify = 'around'),
    
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='table-editing-simple-second',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=12, style= {'font-size': '11px', 'display': 'none'}, align="center")
    ]),
    
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='table-editing-simple-third',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=12, style= {'font-size': '11px'}, align="center")
    ]),
    
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(id='som-metric-table',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=3, style= {'font-size': '11px', 'display': 'none'}, align="center"),
        dbc.Col([
            dash_table.DataTable(id='sales-metric-table',
                                style_header={ 'border': '1px solid black' },
                                style_cell={ 'border': '1px solid grey' },
                                style_table={'overflowX': 'auto'})
        ], width=3, style= {'font-size': '11px', 'display': 'none'}, align="center")
    ]),
    

], fluid = True, style = {'background-color': '#F9F9F9'})


# callback is used to create app interactivity
#@callback()
@callback(
    Output('manufacturer-logo', 'src'),
    Output('channel-logo', 'src'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value')
)
def logo_output(mfg, cnl):
    mfg_src = f'./assets/{mfg}.png'
    cnl_src = f'./assets/{cnl}.png'
    return mfg_src, cnl_src

@callback(
    Output('sensitivity-graph', 'figure'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value')
)
def sensitivity_plot(mfg, cnl):
    global df_feat_imp
    
    if cnl != 'ALL':
        # sensitivity plots
        shap_importance = df_feat_imp[(df_feat_imp['Manufacturer']==mfg)&(df_feat_imp['Channel']==cnl)]
        
        fig_sensitivity = make_subplots()
        fig_sensitivity = fig_sensitivity.add_trace(go.Bar(
                x=shap_importance['feature_importance_vals'],
                y=shap_importance['Feature'],
                orientation='h',
                marker=dict(color =shap_importance['Color'])))

        fig_sensitivity.update_layout(paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                        xaxis_title = 'Importance',
                        yaxis_tickangle=-45,
                        title= f'{mfg} X {cnl} Feature Importance'
                        )
    else:
        fig_sensitivity = go.Figure()
        fig_sensitivity.update_layout(
            xaxis =  { "visible": False },
            yaxis = { "visible": False },
            annotations = [
                {   
                    "text": "Please Select Channel Except ALL",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        )
    return fig_sensitivity


@callback(
    Output('table-editing-simple', 'data'),
    Output('table-editing-simple', 'columns'),
    Input('chosen-manufacturer', 'value'),
    Input('chosen-channel', 'value'),
    Input('reset-button', 'n_clicks'))
def actualize_db(mfg, cnl, n_clicks):
    global display_cols, df_sales_pred, df_all
    if cnl != 'ALL':
        df_slice = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = generate_comp_features_2(df_slice)
        df_slice = df_slice.merge(df_sales_pred[['Manufacturer', 'Channel', 'Date', 'sales_pred', 'SOM_hat']],
                                on = ['Manufacturer', 'Channel', 'Date'])
        df_slice = df_slice.rename(columns= {'sales_pred': 'Predicted Sales', 'SOM_hat': 'Predicted SOM'})
        df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        selected_cols = model_feat_dict[mfg+'_'+cnl]
        fil_sel_cols = [i.split(' Ratio')[0] for i in selected_cols if 'Ratio' in str(i)]
        avg_cols = [i for i in fil_sel_cols if 'PPU' in str(i)]
        if avg_cols and f'{mfg} PPU' not in avg_cols:
            avg_cols = avg_cols + [f'{mfg} PPU']
        tdp_cols = [i for i in fil_sel_cols if 'TDP' in str(i)]
        if tdp_cols and f'{mfg} TDP' not in tdp_cols:
            tdp_cols = tdp_cols + [f'{mfg} TDP']
        selected_cols = [i for i in selected_cols if ('Ratio' not in str(i)) and (str(i).isdigit() == False)]
        selected_cols = list(set(selected_cols + avg_cols + tdp_cols))
        List2 = selected_cols
        selected_cols2 = sorted(List2, key=sort_fun)
        prediction_cols = ['Predicted Sales', 'Predicted SOM']
        prediction_cols = [
            {'name': i, 'id': i, 'editable': False} for i in prediction_cols 
        ]
        list_cols = [
            {'name': i, 'id': i, 'editable': True} for i in selected_cols2 
        ]
        list_cols = display_cols + prediction_cols + list_cols
        cols_to_display = ['Manufacturer', 'Channel', 'Date', 'Sales', 'Predicted Sales', 'SOM', 'Predicted SOM', 'Month'] + selected_cols
        df_slice = df_slice.tail(26)[cols_to_display]
    
    else:
        df_slice = df_all[(df_all['Manufacturer']==mfg)]
        df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_slice = df_slice[['Manufacturer', 'Channel', 'Date', 'Sales',  'SOM', 'sales_pred', 'SOM_hat']]
        df_slice = df_slice.rename(columns= {'sales_pred': 'Predicted Sales', 
                                              'SOM_hat': 'Predicted SOM'})
        df_slice = df_slice.round(decimals= 2)
        df_slice = df_slice.tail(26)
        list_cols = [
            {'name': i, 'id': i, 'editable': False} for i in df_slice.columns 
        ]
    
    if n_clicks > 0:
        if cnl != 'ALL':
            df_slice = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
            df_slice['Date'] = pd.to_datetime(df_slice['Date'])
            df_slice = generate_comp_features_2(df_slice)
            df_slice = df_slice.merge(df_sales_pred[['Manufacturer', 'Channel', 'Date', 'sales_pred', 'SOM_hat']],
                                    on = ['Manufacturer', 'Channel', 'Date'])
            df_slice = df_slice.rename(columns= {'sales_pred': 'Predicted Sales', 'SOM_hat': 'Predicted SOM'})
            df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            selected_cols = model_feat_dict[mfg+'_'+cnl]
            fil_sel_cols = [i.split(' Ratio')[0] for i in selected_cols if 'Ratio' in str(i)]
            avg_cols = [i for i in fil_sel_cols if 'PPU' in str(i)]
            if avg_cols and f'{mfg} PPU' not in avg_cols:
                avg_cols = avg_cols + [f'{mfg} PPU']
            tdp_cols = [i for i in fil_sel_cols if 'TDP' in str(i)]
            if tdp_cols and f'{mfg} TDP' not in tdp_cols:
                tdp_cols = tdp_cols + [f'{mfg} TDP']
            selected_cols = [i for i in selected_cols if ('Ratio' not in str(i)) and (str(i).isdigit() == False)]
            selected_cols = list(set(selected_cols + avg_cols + tdp_cols))
            List2 = selected_cols
            selected_cols2 = sorted(List2, key=sort_fun)
            prediction_cols = ['Predicted Sales', 'Predicted SOM']
            prediction_cols = [
                {'name': i, 'id': i, 'editable': False} for i in prediction_cols 
            ]
            list_cols = [
                {'name': i, 'id': i, 'editable': True} for i in selected_cols2 
            ]
            list_cols = display_cols + prediction_cols + list_cols
            cols_to_display = ['Manufacturer', 'Channel', 'Date', 'Sales', 'Predicted Sales', 'SOM', 'Predicted SOM', 'Month'] + selected_cols
            df_slice = df_slice.tail(26)[cols_to_display]
        
        else:
            df_slice = df_all[(df_all['Manufacturer']==mfg)]
            df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            df_slice = df_slice[['Manufacturer', 'Channel', 'Date', 'Sales',  'SOM', 'sales_pred', 'SOM_hat']]
            df_slice = df_slice.rename(columns= {'sales_pred': 'Predicted Sales', 
                                                'SOM_hat': 'Predicted SOM'})
            df_slice = df_slice.round(decimals= 2)
            df_slice = df_slice.tail(26)
            list_cols = [
                {'name': i, 'id': i, 'editable': False} for i in df_slice.columns 
            ]
        
    return df_slice.to_dict('records'), list_cols

@callback(
    Output("download-dataframe-csv", "data"),
    Output("export-button", "n_clicks"),
    Input("export-button", "n_clicks"),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'),
    prevent_initial_call=True,
)
def download_data(n_clicks, rows, columns):
    if n_clicks > 0:
        df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        n_clicks = 0
        return dcc.send_data_frame(df.to_csv, "simulation_data.csv"), n_clicks
    else:
        n_clicks = 0
        return None, n_clicks

@callback(
    Output('table-editing-simple-second', 'data'),
    Output('table-editing-simple-second', 'columns'),
    Input('table-editing-simple', 'data')
)
def upDate_columns(data):
    global display_cols, df_comp
    df_slice = pd.DataFrame(data)
    Channel = df_slice.Channel.unique()[0]
    if Channel != 'ALL':
        df_slice = df_slice.drop(columns = ['Predicted Sales', 'Predicted SOM'])
        mfg, cnl = df_slice['Manufacturer'].unique()[0], df_slice['Channel'].unique()[0]
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = df_slice.sort_values(by = 'Date')
        
        df_prev = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
        df_prev['Date'] = pd.to_datetime(df_prev['Date'])
        df_prev = df_prev.sort_values(by = 'Date')
        selected_cols = model_feat_dict[mfg+'_'+cnl]
        fil_sel_cols = [i.split(' Ratio')[0] for i in selected_cols if 'Ratio' in str(i)]
        avg_cols = [i for i in fil_sel_cols if 'PPU' in str(i)]
        if avg_cols and f'{mfg} PPU' not in avg_cols:
            avg_cols = avg_cols + [f'{mfg} PPU']
        tdp_cols = [i for i in fil_sel_cols if 'TDP' in str(i)]
        if tdp_cols and f'{mfg} TDP' not in tdp_cols:
            tdp_cols = tdp_cols + [f'{mfg} TDP']
        selected_cols = [i for i in selected_cols if ('Ratio' not in str(i)) and (str(i).isdigit() == False)]
        selected_cols = list(set(selected_cols + avg_cols + tdp_cols))
        cols_to_display = ['Manufacturer', 'Channel', 'Date', 'Sales', 'SOM', 'PPU', 'TDP'] + selected_cols
        df_prev = df_prev.drop(columns = cols_to_display)
        df_prev = df_prev.tail(26)
        df_slice = pd.concat([df_slice, df_prev.set_index(df_slice.index)], axis=1)
        avg_cols = ['COMESTIBLES PPU', 'OTROS PPU', 'PEPSICO PPU', 'RAMO PPU', 'YUPI PPU']
        tdp_cols = ['COMESTIBLES TDP', 'OTROS TDP', 'PEPSICO TDP', 'RAMO TDP', 'YUPI TDP']

        for col in avg_cols:
            df_slice[f'{col} Ratio'] = df_slice[col].astype(float).div(df_slice[f'{mfg} PPU'].astype(float))

        for col in tdp_cols:
            df_slice[f'{col} Ratio'] = df_slice[col].astype(float).div(df_slice[f'{mfg} TDP'].astype(float))
    
    else:
        df_slice = df_slice
    return df_slice.to_dict('records'), [{"name": i, "id": i} for i in df_slice.columns]

@callback(
    Output('table-editing-simple', 'style_data_conditional'),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'),
    Input('chosen-channel', 'value')
)
def style_data_table(rows, columns, cnl):
    global display_cols, df_sales_pred
    if cnl != 'ALL':
        cols = [c['name'] for c in columns]
        cols = cols[8:]
        data = pd.DataFrame(rows, columns=cols)
        index = data.index.tolist()
        
        df_slice = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)]
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice = generate_comp_features_2(df_slice)
        df_slice = df_slice.merge(df_sales_pred[['Manufacturer', 'Channel', 'Date', 'sales_pred', 'SOM_hat']],
                                on = ['Manufacturer', 'Channel', 'Date'])
        df_slice = df_slice.rename(columns= {'sales_pred': 'Predicted Sales', 'SOM_hat': 'Predicted SOM'})
        df_slice['Date'] = df_slice['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        data_previous = df_slice.tail(26)[cols]
        data_previous.index = index
        style_data = diff_dashtable(data, data_previous)
        style_data_format = [{'if': {'row_index': row['row_index'], 'column_id': row['column_id']},
                              'background-color': 'rgba(240,120,54, 0.2)'} for row in style_data]
    else:
        style_data_format = []
    return style_data_format

# For som and sales table
@callback(
    Output('som-graph', 'figure'),
    Output('sales-graph', 'figure'),
    Output('som-ytd', 'children'),
    Output('som-ytd-text', 'children'),
    Output('sales-ytd', 'children'),
    Output('sales-ytd-text', 'children'),
    Output('som-next', 'children'),
    Output('som-next-text', 'children'),
    Output('som-next-year', 'children'),
    Output('som-next-year-text', 'children'),
    Output('sales-next', 'children'),
    Output('sales-next-text', 'children'),
    Output('sales-next-year', 'children'),
    Output('sales-next-year-text', 'children'),
    Output('dist-plot', 'figure'),
    # Output('monthly-dist-plot', 'figure'),
    Input('table-editing-simple-second', 'data'),
    Input('table-editing-simple-second', 'columns'), 
    Input('chosen-channel', 'value'), 
    Input('chosen-manufacturer', 'value'),
    Input('time-selector', 'value'))
def display_output(rows, columns, chnl, mftr, radio_value):
    global df_comp, regression_features, df_sales_pred, max_date, df_all
    
    if chnl != 'ALL':
        df_som_metric = pd.DataFrame(columns= ['Error Metric for March-August', 'MAE (%)'])
        df_sales_metric = pd.DataFrame(columns= ['Error Metric for March-August', 'WAPE (%)', 'R Squared (%)'])
        
        df_rest = df_sales_pred.copy()
        df_slice = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        df_slice['Date'] = pd.to_datetime(df_slice['Date'])
        df_slice['Month'] = df_slice['Month'].astype(int)
        
        assert df_slice['Manufacturer'].nunique()==1
        assert df_slice['Channel'].nunique()==1
        mfg = df_slice['Manufacturer'].unique()[0]
        cnl = df_slice['Channel'].unique()[0]
        
        df_slice_whole = df_comp[(df_comp['Manufacturer']==mfg)&(df_comp['Channel']==cnl)].iloc[:-26]
        df_slice_whole['Date'] = pd.to_datetime(df_slice_whole['Date'])
        df_slice_whole = generate_comp_features_2(df_slice_whole)
        df_slice_whole['Month'] = df_slice_whole['Month'].astype(int)
        df_slice_whole = df_slice_whole.sort_values(by='Date')
        columns = df_slice.columns.tolist()
        df_slice_whole = pd.concat([df_slice_whole[columns], df_slice])
        encoded_features = pd.get_dummies(df_slice_whole['Month'])
        df_slice2 = pd.concat([df_slice_whole,encoded_features],axis=1)
        
        df_rest = df_rest[~((df_rest['Manufacturer']==mfg)&(df_rest['Channel']==cnl))][columns + ['sales_pred']]
        df_rest['Date'] = pd.to_datetime(df_rest['Date'])
        df_rest['sales_pred'] = df_rest['sales_pred'].astype(float)
        model = model_dict.get(mfg+cnl)
        model_features = model_feat_dict.get(mfg+"_"+cnl)
        df_slice2[regression_features] = df_slice2[regression_features].astype(float)
        df_slice2[regression_features] = df_slice2[regression_features].apply(np.log10)
        df_slice2['sales_pred'] = model.predict(df_slice2[model_features])

        df_slice_whole['sales_pred'] = model.predict(df_slice2[model_features])
        
        # Driver analysis
        df_driver = df_slice2[['Date', 'Month', 'Sales', 'sales_pred']+regression_features+encoded_features.columns.tolist()]

        df_driver = df_driver.sort_values(['Date'])
        df_driver = df_driver[['Date', 'Month', 'Sales','sales_pred'] + model_features]

        sel_reg_features = list(set(regression_features).intersection(set(model_features)))
        sel_oh_cols = list(set(model_features).difference(set(regression_features)))
        
        weights = model.coef_[0]
        df_driver[model_features] = np.multiply(df_driver[model_features], weights)
        df_driver['intercept'] = model.intercept_[0]
        df_driver['total_sum'] = df_driver[model_features + ['intercept']].sum(axis = 1)
        df_driver['sum'] = df_driver[model_features].abs().sum(axis = 1)
        for feature in model_features:
            df_driver[feature] = (df_driver[feature])/df_driver['sum']
        
        df_driver['Month'] = df_driver[sel_oh_cols].sum(axis = 1)
        df_driver = df_driver.drop(columns= sel_oh_cols)

        fig_driver = make_subplots(specs=[[{"secondary_y": True}]])
        disp_features = list(set(df_driver.columns[4:].tolist()).difference(set(['total_sum', 'sum', 'intercept']))) + ['Month']

        for kpi in disp_features:
            fig_driver.add_trace(go.Bar(x=df_driver['Date'], y=df_driver[kpi],
                                name=kpi), 
                        secondary_y=False)
        fig_driver.update_layout(barmode='stack', showlegend=True, template= 'plotly_white',
                                legend= dict(orientation= 'h'))
        
        
        df_whole = pd.concat([df_slice_whole, df_rest])
        df_whole['total_sales_hat'] = df_whole.groupby(['Date', 'Channel'])['sales_pred'].transform('sum')
        df_whole['SOM_hat'] = df_whole['sales_pred']*100/df_whole['total_sales_hat']
        
        # Som and sales card outputs
        card_outputs = df_whole[(df_whole['Manufacturer']==mfg)&(df_whole['Channel']==cnl)]
        card_outputs['year'] = card_outputs['Date'].dt.year
        next_month = (max_date + relativedelta(months=1)).strftime('%b-%Y')
        current_year = (max_date + relativedelta(months=1)).year
        som_next_text, sales_next_text = f'Estimated SOM For {next_month}', f'Estimated Sales For {next_month} (10^6 CoP)'
        som_next_year_text, sales_next_year_text = f'Estimated SOM For Total {current_year}', f'Estimated Sales For Total {current_year} (10^6 CoP)'
        som_next, sales_next = (np.round(card_outputs[card_outputs['Date']==next_month]['SOM_hat'].values[0], 2), 
                                np.int(card_outputs[card_outputs['Date']==next_month]['sales_pred'].values[0]))
        som_next, sales_next = f'{som_next} %', '{:,}'.format(sales_next)
        sales_next_year = card_outputs[card_outputs['year']==current_year]
        sales_next_year['sales_final'] = np.where(sales_next_year['Date']>max_date, sales_next_year['sales_pred'], 
                                                sales_next_year['Sales'])
        sales_next_year['sales_final'] = np.where(sales_next_year['sales_final'].isnull(), sales_next_year['sales_pred'], 
                                                sales_next_year['sales_final'])
        sales_next_year = np.int(sales_next_year[sales_next_year['year']==current_year]['sales_final'].sum())
        sales_next_year = '{:,}'.format(sales_next_year)
        
        som_next_year = df_whole.copy()
        som_next_year['year'] = som_next_year['Date'].dt.year
        som_next_year = som_next_year[som_next_year['year']==current_year]
        som_next_year['sales_final'] = np.where(som_next_year['Date']>max_date, som_next_year['sales_pred'], 
                                                som_next_year['Sales'])
        som_next_year['sales_final'] = np.where(som_next_year['sales_final'].isnull(), som_next_year['sales_pred'], 
                                                som_next_year['sales_final'])
        som_next_year = som_next_year.groupby(['Manufacturer', 'Channel'])['sales_final'].sum().reset_index()
        som_next_year['sales_final_hat'] = som_next_year.groupby(['Channel'])['sales_final'].transform('sum')
        som_next_year['SOM_hat'] = som_next_year['sales_final']*100/som_next_year['sales_final_hat']
        som_next_year = np.round(som_next_year[(som_next_year['Manufacturer']==mfg)&(som_next_year['Channel']==cnl)]['SOM_hat'].values[0], 2)
        som_next_year = f'{som_next_year} %'
        
        # YTD sales and som calculation
        ytd = df_whole.copy()
        ytd['year'] = ytd['Date'].dt.year
        ytd = ytd[(ytd['Date']<= max_date)&(ytd['year']==max_date.year)]
        ytd = ytd.groupby(['Manufacturer', 'Channel'])['Sales'].sum().reset_index()
        ytd['sales_sum'] = ytd.groupby(['Channel'])['Sales'].transform('sum')
        ytd['SOM'] = ytd['Sales']*100/ytd['sales_sum']
        som_ytd = np.round(ytd[(ytd['Manufacturer']==mfg)&(ytd['Channel']==cnl)]['SOM'].values[0], 2)
        som_ytd = f'{som_ytd} %'
        som_ytd_text, sales_ytd_text = f'SOM YTD {current_year}', f'Sales YTD {current_year} (10^6 CoP)'
        sales_ytd = np.int(ytd[(ytd['Manufacturer']==mfg)&(ytd['Channel']==cnl)]['Sales'].values[0])
        sales_ytd = '{:,}'.format(sales_ytd)
        
        ## Sales and SOM plots
        df_plot = df_whole[(df_whole['Manufacturer']==mfg)&(df_whole['Channel']==cnl)]
        df_plot2 = df_plot[df_plot['Date']<=max_date]
        df_plot['SOM'] = np.where(df_plot['Date']==max_date+ relativedelta(months=1), 
                                df_plot['SOM_hat'], df_plot['SOM'])
        df_plot['SOM_hat'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['SOM_hat'])
        df_plot['Sales'] = np.where(df_plot['Date']==max_date + relativedelta(months=1), 
                                    df_plot['sales_pred'], df_plot['Sales'])
        df_plot['sales_pred'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['sales_pred'])
        
        df_plot = df_plot.rename(columns= {'SOM_hat': 'Predicted SOM', 
                                        'sales_pred': 'Predicted Sales'})

        fig_som = go.Figure()
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['SOM'], name= 'SOM', line=dict(color='#015CB4')))
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted SOM'], 
                                name= 'Predicted SOM', marker= dict(color= '#C9002B')))
        fig_som.add_vrect(x0=df_plot.tail(18)['Date'].min(), 
                x1=df_plot.tail(18)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_som.update_layout(title= 'Share of Market Trend (Historical & Predicted)', yaxis_title = 'SOM (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))
        
        fig_sales = go.Figure()
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Sales'], name= 'Sales', line=dict(color='#015CB4')))
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted Sales'], 
                                name= 'Predicted Sales', marker= dict(color= '#C9002B')))
        fig_sales.add_vrect(x0=df_plot.tail(18)['Date'].min(), 
                x1=df_plot.tail(18)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_sales.update_layout(title= 'Market Sales Trend (Historical & Predicted)', yaxis_title = 'Sales (10^6 CoP)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))
        
        # yearly dist plot
        df_yearly = df_whole.copy()
        df_yearly = df_yearly.rename(columns= {'sales_pred': 'Predicted Sales', 
                                               'SOM_hat': 'Predicted SOM'})
        if radio_value == 'Yearly':
            # yearly dist plot
            fig_dist = plot_dist_charts(df_yearly, mfg)
        else:
            # monthly dist plot
            fig_dist = plot_monthly_dist_chart(df_yearly, mfg, current_year)
        
    
    else:
        df_plot = df_all[df_all['Manufacturer']==mftr]
        df_plot['SOM'] = np.where(df_plot['Date']==max_date+ relativedelta(months=1), 
                                df_plot['SOM_hat'], df_plot['SOM'])
        df_plot['SOM_hat'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['SOM_hat'])
        df_plot['Sales'] = np.where(df_plot['Date']==max_date + relativedelta(months=1), 
                                    df_plot['sales_pred'], df_plot['Sales'])
        df_plot['Sales'] = np.where(df_plot['Date']>max_date + relativedelta(months=1), 
                                    np.nan, df_plot['Sales'])
        df_plot['sales_pred'] = np.where(df_plot['Date']<=max_date, np.nan, df_plot['sales_pred'])
        
        df_plot = df_plot.rename(columns= {'SOM_hat': 'Predicted SOM', 
                                        'sales_pred': 'Predicted Sales'})
        df_plot = df_plot[['Manufacturer', 'Channel', 'Date', 'Sales',  'SOM', 'Predicted Sales', 'Predicted SOM']]

        fig_som = go.Figure()
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['SOM'], name= 'SOM', line=dict(color='#015CB4')))
        fig_som.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted SOM'], 
                                name= 'Predicted SOM', marker= dict(color= '#C9002B')))
        fig_som.add_vrect(x0=df_plot.tail(18)['Date'].min(), 
                x1=df_plot.tail(18)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_som.update_layout(title= 'Share of Market Trend (Historical & Predicted)', yaxis_title = 'SOM (%)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))
        
        fig_sales = go.Figure()
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Sales'], name= 'Sales', line=dict(color='#015CB4')))
        fig_sales.add_trace(go.Scatter(x= df_plot['Date'], y = df_plot['Predicted Sales'], 
                                name= 'Predicted Sales', marker= dict(color= '#C9002B')))
        fig_sales.add_vrect(x0=df_plot.tail(18)['Date'].min(), 
                x1=df_plot.tail(18)['Date'].max(),
                line_width=0, fillcolor="blue", opacity=0.1)
        fig_sales.update_layout(title= 'Market Sales Trend (Historical & Predicted)', yaxis_title = 'Sales (10^6 CoP)', paper_bgcolor= '#F9F9F9', plot_bgcolor= '#F9F9F9',
                            legend= dict(orientation= 'h', title=None))

        # Som and sales card outputs
        df_whole = df_all.copy()
        card_outputs = df_whole[(df_whole['Manufacturer']==mftr)]
        card_outputs['year'] = card_outputs['Date'].dt.year
        next_month = (max_date + relativedelta(months=1)).strftime('%b-%Y')
        current_year = (max_date + relativedelta(months=1)).year
        som_next_text, sales_next_text = f'Estimated SOM For {next_month}', f'Estimated Sales For {next_month} (10^6 CoP)'
        som_next_year_text, sales_next_year_text = f'Estimated SOM For Total {current_year}', f'Estimated Sales For Total {current_year} (10^6 CoP)'
        som_next, sales_next = (np.round(card_outputs[card_outputs['Date']==next_month]['SOM_hat'].values[0], 2), 
                                np.int(card_outputs[card_outputs['Date']==next_month]['sales_pred'].values[0]))
        som_next, sales_next = f'{som_next} %', '{:,}'.format(sales_next)
        sales_next_year = card_outputs[card_outputs['year']==current_year]
        sales_next_year['sales_final'] = np.where(sales_next_year['Date']>max_date, sales_next_year['sales_pred'], 
                                                sales_next_year['Sales'])
        sales_next_year['sales_final'] = np.where(sales_next_year['sales_final'].isnull(), sales_next_year['sales_pred'], 
                                                sales_next_year['sales_final'])
        sales_next_year = np.int(sales_next_year[sales_next_year['year']==current_year]['sales_final'].sum())
        sales_next_year = '{:,}'.format(sales_next_year)
        
        som_next_year = df_whole.copy()
        som_next_year['year'] = som_next_year['Date'].dt.year
        som_next_year = som_next_year[som_next_year['year']==current_year]
        som_next_year['sales_final'] = np.where(som_next_year['Date']>max_date, som_next_year['sales_pred'], 
                                                som_next_year['Sales'])
        som_next_year['sales_final'] = np.where(som_next_year['sales_final'].isnull(), som_next_year['sales_pred'], 
                                                som_next_year['sales_final'])
        som_next_year = som_next_year.groupby(['Manufacturer', 'Channel'])['sales_final'].sum().reset_index()
        som_next_year['sales_final_hat'] = som_next_year.groupby(['Channel'])['sales_final'].transform('sum')
        som_next_year['SOM_hat'] = som_next_year['sales_final']*100/som_next_year['sales_final_hat']
        som_next_year = np.round(som_next_year[(som_next_year['Manufacturer']==mftr)&(som_next_year['Channel']=='ALL')]['SOM_hat'].values[0], 2)
        som_next_year = f'{som_next_year} %'
        
        # YTD sales and som calculation
        ytd = df_whole.copy()
        ytd['year'] = ytd['Date'].dt.year
        ytd = ytd[(ytd['Date']<= max_date)&(ytd['year']==max_date.year)]
        ytd = ytd.groupby(['Manufacturer', 'Channel'])['Sales'].sum().reset_index()
        ytd['sales_sum'] = ytd.groupby(['Channel'])['Sales'].transform('sum')
        ytd['SOM'] = ytd['Sales']*100/ytd['sales_sum']
        som_ytd = np.round(ytd[(ytd['Manufacturer']==mftr)&(ytd['Channel']=='ALL')]['SOM'].values[0], 2)
        som_ytd = f'{som_ytd} %'
        som_ytd_text, sales_ytd_text = f'SOM YTD {current_year}', f'Sales YTD {current_year} (10^6 CoP)'
        sales_ytd = np.int(ytd[(ytd['Manufacturer']==mftr)&(ytd['Channel']=='ALL')]['Sales'].values[0])
        sales_ytd = '{:,}'.format(sales_ytd)
        
        # distribution charts
        fig_dist = go.Figure()
        fig_dist.update_layout(
            xaxis =  { "visible": False },
            yaxis = { "visible": False },
            annotations = [
                {   
                    "text": "Please Select Channel Except ALL",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        )
    

    return (fig_som, fig_sales, 
           som_ytd, som_ytd_text, sales_ytd, sales_ytd_text, som_next, som_next_text, som_next_year, 
            som_next_year_text, sales_next, sales_next_text, sales_next_year, sales_next_year_text, fig_dist
            )
