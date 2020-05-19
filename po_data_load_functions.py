# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:34:40 2019

@author: Администратор
"""
import pandas as pd
import numpy as np
from os import path as os_path
import sys
folderSnippets = r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\standard_functions'
sys.path.insert(0,folderSnippets)
from market_standard_functions import fix_constr_conting_dir
sys.path.insert(0, r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\risk_management")
from risk_management import add_constr_influence_columns
sys.path.insert(0, r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\pathsOptimizer")
from po_system_functions import get_common_root

def get_constr_table_data(iso, class_mode, constraints_table_file):
    
    df_constraints_table = pd.read_csv(constraints_table_file)
     # remove blank rows
    df_constraints_table.dropna(how='all',inplace=True)
    df_constraints_table = df_constraints_table[~df_constraints_table.ConstraintDevice.fillna('').str.contains('"')]
    
    df_constraints_table = fix_constr_conting_dir(iso, df_constraints_table)

    dictMainDev = {'MainDev':'Dev', 'Main':'MainConstrStrong','MainStrong':'MainConstrStrong', 'MainWeak':'MainConstrWeak'}    
    df_constraints_table.Main.replace(dictMainDev, inplace=True)

    if (df_constraints_table.Possibility[df_constraints_table.Main.fillna('').str.contains('MainConstr')] < 70).any():
        print('\n!!!!!!!!\n\nWARNING!! Low possibility (< 70) for main constr please check it!\n\n!!!!!!!!\n')
        #raise ValueError('Low possibility (< 70) for main constr please check it!')
    
    # filter peak or off-peak, peak we
    if class_mode == 1: # peak case
        df_constraints_table = df_constraints_table[(~df_constraints_table.Class.str.contains('off',case=False)) & (~df_constraints_table.Class.str.contains('we',case=False))] 
    elif class_mode == 0: # off-peak case
        df_constraints_table = df_constraints_table[df_constraints_table.Class.str.contains('off',case=False)]
    elif class_mode == 2: # peak we case
        df_constraints_table = df_constraints_table[df_constraints_table.Class.str.contains('we',case=False)]

    if (iso == "ERCOT") | (iso == "NYISO") | (iso == "ISONE"):
        df_constraints_table.Direction = 1
    
    # set blank contingencies to 'BASE'
    df_constraints_table.loc[df_constraints_table.ContingencyDevice.isnull(),'ContingencyDevice'] = 'BASE'

    # fix direction
    df_constraints_table['Direction'] = pd.to_numeric(df_constraints_table['Direction'].fillna(0), errors='ignore', downcast='integer')
    df_constraints_table.Direction = df_constraints_table.Direction.astype(str)
    # create Key Constraint,Contingency,Direction
    df_constraints_table['conID'] = df_constraints_table.ConstraintDevice.fillna('') + '+' + df_constraints_table.ContingencyDevice.fillna('') + '+' + df_constraints_table.Direction
    # remove blank rows
    df_constraints_table = df_constraints_table[df_constraints_table.conID.notnull()]
    df_constraints_table.sort_values('Main').drop_duplicates('conID', inplace=True)
            
    # fill empty data with '' and 0
    df_constraints_table[['ConstraintDevice','ContingencyDevice']] = df_constraints_table[['ConstraintDevice','ContingencyDevice']].fillna('')
    df_constraints_table[['N-','N+','MonthSumMin','MonthSumMax','Possibility']] = df_constraints_table[['N-','N+','MonthSumMin','MonthSumMax','Possibility']].fillna(0)
    df_constraints_table[['Main']] = df_constraints_table[['Main']].fillna('')
    df_constraints_table[['OldStatus']] = df_constraints_table[['OldStatus']].fillna('')
    df_constraints_table[['DisappearStatus']] = df_constraints_table[['DisappearStatus']].fillna('')
    df_constraints_table[['Seasonality']] = df_constraints_table[['Seasonality']].fillna('')

    # only used for output file (Comment column if constraints MAIN or not)
    df_constraints_table['For_Comment'] = df_constraints_table.Main
    df_constraints_table.loc[df_constraints_table.OldStatus.notnull(),'For_Comment'] = df_constraints_table.For_Comment+' '+df_constraints_table.OldStatus
    df_constraints_table.loc[df_constraints_table.For_Comment.isnull(),'For_Comment'] = df_constraints_table.OldStatus
    
    return df_constraints_table

def get_RC_tool_data(file_RCTool):
    
    # read, clear file
    df_points_corr = pd.read_csv(file_RCTool, names='Point1,Point2,corDA50,corDA100,corDA150,corRT50,corRT100,corRT150'.split(','))
    
    df_points_corr = df_points_corr.applymap(lambda x: x.strip() if type(x) is str else x)
    # clear from { }
    df_points_corr.Point1 = df_points_corr.Point1.str.replace('{{"','')
    df_points_corr.Point1 = df_points_corr.Point1.str.replace('"}','')
    df_points_corr.Point2 = df_points_corr.Point2.str.replace('"','')
    df_points_corr.corRT150 = df_points_corr.corRT150.str.replace('}','')
    # replace NO DATA and empty data with 0
    df_points_corr.replace('"NO DATA"',np.nan,inplace=True)
    for col_ in ['corDA50','corDA100','corDA150','corRT50','corRT100','corRT150']:
        df_points_corr[col_] = df_points_corr[col_].fillna(0).astype(str).str.replace('*^','E',regex=False)
        # fix not numeric values in columns by pd.to_numeric, astype(float) not work in this case if word in values
        df_points_corr[col_] = pd.to_numeric(df_points_corr[col_], errors='coerce', downcast='float')
   
    
    return df_points_corr
    
def get_source_sink_file_data(iso, class_mode, file_source_sink, IS_REMOVE_HUBS):

    df_points_sourcesink_source_f = pd.read_csv(file_source_sink,encoding ='latin1')
    if (iso == 'SPP'):
        df_points_sourcesink_source_f.fillna(method='ffill', inplace=True)
    if (iso == 'CAISO'):
        new_cols = df_points_sourcesink_source_f.columns.str.strip('ns1:')
        df_points_sourcesink_source_f.rename(columns=dict(zip(df_points_sourcesink_source_f.columns, new_cols)),inplace=True)
        df_points_sourcesink_source_f.rename(columns={'TimeOfUse':'Class'},inplace=True)
    if (iso == 'NYISO'):
        df_points_sourcesink_source_f.rename(columns={'PTID Name':'Name','Bus No.':'BusName'},inplace=True)
    if (iso == 'ISONE'):
        df_points_sourcesink_source_f.rename(columns={'Node Name':'Name'},inplace=True)
        # for ISONE also remove spaces between parts of node name: "aaaa    bb" ->"aaaa bb"
        df_points_sourcesink_source_f.Name = df_points_sourcesink_source_f.Name.str.strip().str.split().str.join(" ")
        
    # drop duplicated
    if (class_mode == 1) & ('Class' in df_points_sourcesink_source_f.columns): # peak case
        df_points_sourcesink_source = df_points_sourcesink_source_f[(~df_points_sourcesink_source_f.Class.str.contains('off',case=False)) & (~df_points_sourcesink_source_f.Class.str.contains('we',case=False))]
    elif (class_mode == 0) & ('Class' in df_points_sourcesink_source_f.columns): # off peak case
        df_points_sourcesink_source = df_points_sourcesink_source_f[df_points_sourcesink_source_f.Class.str.contains('off',case=False)]
    elif (class_mode == 2) & ('Class' in df_points_sourcesink_source_f.columns): # peak we case
        df_points_sourcesink_source = df_points_sourcesink_source_f[df_points_sourcesink_source_f.Class.str.contains('we',case=False)]
    else: # if there is no Class column
        df_points_sourcesink_source = df_points_sourcesink_source_f 
        
    if IS_REMOVE_HUBS:
        df_points_sourcesink_source_gr = df_points_sourcesink_source.groupby(['Name']).agg({'PriceNode':'count'}).reset_index()
        df_points_sourcesink_source_gr = df_points_sourcesink_source_gr[df_points_sourcesink_source_gr.PriceNode == 1]
        dict_not_hubs = pd.Series(1, index=df_points_sourcesink_source_gr.Name).to_dict()
        df_points_sourcesink_source['N_buses'] = df_points_sourcesink_source.Name.map(dict_not_hubs)
        df_points_sourcesink_source = df_points_sourcesink_source[(df_points_sourcesink_source.N_buses.notnull())]
        file_save_source_sinks_no_hubs = os_path.join(os_path.dirname(file_source_sink),'SourceSink_NO_HUBS.csv')
        df_points_sourcesink_source.to_csv(file_save_source_sinks_no_hubs, index=0)
    
    return df_points_sourcesink_source

def get_shadow_price_data(iso, class_mode, file_shadow_price, IS_ANNUAL_PRICES):
    # shadow price from auction
    df_shadow_price = pd.read_csv(file_shadow_price)
    if iso == 'CAISO':
        df_shadow_price=df_shadow_price.iloc[1:]
    df_shadow_price.columns = df_shadow_price.columns.str.strip()
    if iso == 'SPP':
        df_shadow_price.rename(columns={'ClearingPrice':'ShadowPrice'},inplace=True)
     
    if iso == 'ERCOT':    
        df_shadow_price.rename(columns={'ShadowPricePerMWH':'ShadowPrice', 'TimeOfUse':'Class'},inplace=True)
    
    if iso == 'CAISO':
        df_shadow_price.rename(columns={'Node':'SourceSink'},inplace=True)
    
    if IS_ANNUAL_PRICES:
        df_shadow_price.rename(columns={'Node':'SourceSink'},inplace=True)
        
    if class_mode == 1:
        if iso == 'CAISO':
            df_shadow_price.rename(columns={'LMP.1':'ShadowPrice'},inplace=True)
        else:
            df_shadow_price = df_shadow_price[(~df_shadow_price.Class.str.contains('off',case=False)) & (~df_shadow_price.Class.str.contains('we',case=False))]
        
    elif class_mode == 0:
        if iso == 'CAISO':
            df_shadow_price.rename(columns={'LMP.2':'ShadowPrice'},inplace=True)
        else:
            df_shadow_price = df_shadow_price[df_shadow_price.Class.str.contains('off',case=False)]
    elif class_mode == 2:
        df_shadow_price = df_shadow_price[df_shadow_price.Class.str.contains('we',case=False)]
        
    return df_shadow_price

def get_paths_to_use(SOURCE_SINK_TO_USE_FILE, list_combination):
     # use given sources and sink or remove needed
    if  os_path.isfile(SOURCE_SINK_TO_USE_FILE):
        df_source_sink_to_use = pd.read_csv(SOURCE_SINK_TO_USE_FILE)
        
        # case when only given points should be used
        if (df_source_sink_to_use.Operation == "Use").all():
            sources_to_use = list(df_source_sink_to_use[df_source_sink_to_use.Type == 'Source'].Node)
            sinks_to_use = list(df_source_sink_to_use[df_source_sink_to_use.Type == 'Sink'].Node)
            list_combination = [[source, sink] for source, sink in list_combination if source in sources_to_use and sink in sinks_to_use]
        # case when given points should be removed
        elif (df_source_sink_to_use.Operation == "Remove").all():
            sources_to_remove = list(df_source_sink_to_use[df_source_sink_to_use.Type == 'Source'].Node)
            sinks_to_remove = list(df_source_sink_to_use[df_source_sink_to_use.Type == 'Sink'].Node)
            list_combination = [[source, sink] for source, sink in list_combination if source not in sources_to_remove and sink not in sinks_to_remove]
        # mix case or misprints in Operation column - both wrong cases
        else:
            raise ValueError("Please check that ALL data in Operation column is \"Remove\" or \"Use\" but not mixed!")
    else: # no file case
        print('Can not find '+SOURCE_SINK_TO_USE_FILE+' file')        
    
    return list_combination
    
def get_start_portfolio(portfolio_file, class_mode, list_constraints, dict_coeff_constraints, dict_rename, dict_points_influence, log_path):
    print("Loading existing portfolio from {}".format(portfolio_file))
    df_start_portfolio = pd.read_csv(portfolio_file)
    
    # load only peak or off-peak data
    if ('Class' in df_start_portfolio.columns) & (class_mode == 1): # peak case
        df_start_portfolio = df_start_portfolio[(~df_start_portfolio.Class.str.contains('off',case=False)) & (~df_start_portfolio.Class.str.contains('we',case=False))]
    elif ('Class' in df_start_portfolio.columns) & (class_mode == 0): # off-peak case
        df_start_portfolio = df_start_portfolio[df_start_portfolio.Class.str.contains('off',case=False)]
    elif ('Class' in df_start_portfolio.columns) & (class_mode == 2): # peak we case
        df_start_portfolio = df_start_portfolio[df_start_portfolio.Class.str.contains('we',case=False)]
    
    df_start_portfolio = df_start_portfolio.groupby(['Source','Sink', 'Price','Portfolio']).agg({'MWs':sum}).reset_index()
    
    df_start_portfolio = add_constr_influence_columns(list_constraints, df_start_portfolio, dict_points_influence, log_path)
    
    df_start_portfolio['FV'] = 0

    df_start_portfolio['Collateral'] = df_start_portfolio['Collateral_t'] = (df_start_portfolio.Portfolio/df_start_portfolio.MWs).astype(np.int32)
    
    df_start_portfolio['flagStop'] = 11
    df_start_portfolio.flagStop = df_start_portfolio.flagStop.astype(np.int8)
    
    df_start_portfolio['Capital'] = (df_start_portfolio.MWs*df_start_portfolio.Price).astype(np.float32)
       
    # set corr points, duplicated above, but for check
    df_start_portfolio['Source1'] = df_start_portfolio.Source
    df_start_portfolio['Sink1'] = df_start_portfolio.Sink
    
    df_start_portfolio['Take_constraints'] = '' # columns with constr names, delim ;
    df_start_portfolio['Take_constraints_byOneNode'] = ''
    df_start_portfolio['is_non_zero_infl'] = 0
    
    # fill mw_constr columns and correct  dict_coeff_constraints if its already exceeds limits
    i = 0
    for constr in list_constraints:
        if constr in df_start_portfolio.columns:
            df_start_portfolio['mws_'+constr] = df_start_portfolio[constr]*df_start_portfolio.MWs  
            
            # correct N- N+ for avg coef of start portfolio
            coef_portfolio_neg = df_start_portfolio['mws_'+constr].where(lambda x: x<0).fillna(0).sum()
            # dict_coeff_constraints[0] = [N-,N+]
            N_minus = dict_coeff_constraints[constr][0][0]
            # if  coef_portfolio < N- -> make N- = coef_portfolio
            if coef_portfolio_neg < N_minus:
                dict_coeff_constraints[constr][0][0] = coef_portfolio_neg-0.01
                
        if i%100 ==0:
            print('Load portfolio: {} constr done'.format(i))
        i += 1
    
    log_file = get_common_root(log_path,r'Portfolio_start.csv')
    df_start_portfolio.to_csv(log_file, index=0, float_format='%.3f')
    
    return (df_start_portfolio, dict_coeff_constraints)

    
# =============================================================================
#     # use clusters data to restore source sink names as they named in df_points 
#     df_start_portfolio['SourceSink'] = df_start_portfolio.Source.map(lambda x: dict_rename.get(x,x))+df_start_portfolio.Sink.map(lambda x: dict_rename.get(x,x))
#     
#     # get dict for MWs
#     dict_portfolio_paths_mws = pd.Series(df_start_portfolio.MWs.values,index=df_start_portfolio.SourceSink).to_dict()    
#     
#     # get dict for price
#     dict_portfolio_paths_price = pd.Series(df_start_portfolio.Price.values,index=df_start_portfolio.SourceSink).to_dict()
#       
#     # choose df_points data with paths of start portfolio
#     df_points_start = df_points[(df_points.Source+df_points.Sink).isin(df_start_portfolio['SourceSink'].to_list())].copy()
#     
#     df_points_start.MWs = (df_points_start.Source+df_points_start.Sink).map(dict_portfolio_paths_mws).fillna(0).astype(np.int8)
#     df_points_start.Price = (df_points_start.Source+df_points_start.Sink).map(dict_portfolio_paths_price).fillna(0).astype(np.float32)    
#     df_points_start['MWs_start'] = df_points_start.MWs
# =============================================================================
