# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:55:18 2019

@author: Администратор
"""
import pandas as pd
import numpy as np
from os import path as os_path
from copy import deepcopy as copy_deepcopy
import sys
sys.path.insert(0, r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\pathsOptimizer")
from po_system_functions import LogStageValue, get_common_root
from po_data_load_functions import get_RC_tool_data
from collections import defaultdict
from datetime import timedelta
import glob
folderSnippets = r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\standard_functions'
sys.path.insert(0,folderSnippets)
from get_path_to import get_path_to_node_history_lib

def calc_Price(df_points, class_mode, CLUSTERS_iso, CLUSTERS_price_distribution_file, log_path, coef_sigma_down, ignor_3minFTR_for_down_price, df_shadow_price,IS_CALC_PRICE_FROM_MEAN_SIGMA,  min_price_delta, min_price_fraction_delta, min_add_price, price_percent, min_value_price, IS_CALC_PRICE_FROM_REF_PRICE = False, IS_OPTION = False, IS_ANNUAL_PRICES = False, ANNUAL_PRICE_MODE = '', IS_DIVIDE_QUARTAL_PRICES = False, IS_DOWN_PRICE=False, USE_MIN_RT_DA=False, min_RT=0, min_DA=0):
    
    if IS_CALC_PRICE_FROM_MEAN_SIGMA:
        df_points = calc_Price_by_mean_and_sigma(df_points, class_mode, CLUSTERS_price_distribution_file, min_price_delta, min_price_fraction_delta, log_path, coef_sigma_down, ignor_3minFTR_for_down_price, IS_OPTION, IS_DOWN_PRICE, USE_MIN_RT_DA, min_RT, min_DA)  
    elif IS_CALC_PRICE_FROM_REF_PRICE:
        df_points = calc_Price_from_ref_price(df_points, log_path)      
    else:
        df_points = calc_Price_from_node_price_file(df_points, df_shadow_price, IS_ANNUAL_PRICES, ANNUAL_PRICE_MODE)
    
    if IS_DIVIDE_QUARTAL_PRICES:
        df_points.Price = df_points.Price/3
        df_points.Price = df_points.Price.astype(np.float64)
    
    if (CLUSTERS_iso == "ERCOT") & (not IS_CALC_PRICE_FROM_MEAN_SIGMA):
        df_points.Price = df_points.Price*352
        df_points.Price = df_points.Price.astype(np.float64)
    
    df_points['Price_t'] = df_points.Price
    
    if min_add_price & (not IS_CALC_PRICE_FROM_REF_PRICE):
        df_points.loc[df_points.Price*price_percent > min_value_price,'Price'] = df_points.Price+df_points.Price.abs()*price_percent
        df_points.loc[df_points.Price*price_percent <= min_value_price,'Price'] = df_points.Price + min_value_price    
    
    return df_points

def calc_Price_from_ref_price(df_points,log_path, CLUSTER_ref_price_file="", min_FV=200, FV_percent_for_price=0.25, max_price=300):
    
    df_ref_price = pd.read_csv(CLUSTER_ref_price_file, sep='|')
    df_ref_price['NC'] = df_ref_price.SOURCE_LOCATION + df_ref_price.SINK_LOCATION
    dict_ref_price = pd.Series(df_ref_price.PRODUCT_REFERENCE_PRICE.fillna(0).values,index=df_ref_price.NC).to_dict()
    df_points['NC'] = df_points.Source+df_points.Sink
    df_points['Ref_price'] = df_points.NC.map(dict_ref_price).fillna(0)
    df_points = df_points[df_points.Ref_price > 0] 
    df_points = df_points[df_points.FV > min_FV] 
    
    df_points['Price_from_FV'] = FV_percent_for_price*df_points.FV
    df_points['Price'] = df_points[['Price_from_FV','Ref_price']].min(axis=1)
    df_points = df_points[df_points.Price <= max_price] 
    
    return df_points

def calc_Price_by_mean_and_sigma(df_points, class_mode, CLUSTERS_price_distribution_file, min_price_delta, min_price_fraction_delta, log_path, coef_sigma_down, ignor_3minFTR_for_down_price, IS_OPTION, IS_DOWN_PRICE=False, USE_MIN_RT_DA=False, min_RT=0, min_DA=0):
    # read file with mean and sigma of path price
    df_distr_price = pd.read_csv(CLUSTERS_price_distribution_file)
    if class_mode == 1:
        dict_rename_header = {'Pr_Peak':'PredictedPrice', 'dev+ Peak':'StDevPlus','dev- Peak':'StDevMinus','dev_Peak':'StDev', 'DA_Peak':'DA', 'minDA/RT_Peak':'minDART', 'RT_Peak':'RT', '2minFTR_Peak':'3minFTR', 'DA_week_Peak':'WeekDA', 'RT_week_Peak':'WeekRT'}
    elif class_mode == 0: #off peak case
        dict_rename_header = { 'Pr_OffPeak':'PredictedPrice', 'dev+ OffPeak':'StDevPlus','dev- OffPeak':'StDevMinus','dev_OffPeak':'StDev', 'DA_OffPeak':'DA', 'minDA/RT_OffPeak':'minDART','RT_OffPeak':'RT', '2minFTR_OffPeak':'3minFTR', 'DA_week_OffPeak':'WeekDA', 'RT_week_OffPeak':'WeekRT'}
    elif class_mode == 2: # peak we case
        dict_rename_header = { 'Pr_PeakWE':'PredictedPrice', 'dev+ PeakWE':'StDevPlus','dev- PeakWE':'StDevMinus', 'dev_PeakWE':'StDev', 'DA_PeakWE':'DA', 'minDA/RT_PeakWE':'minDART','RT_PeakWE':'RT', '2minFTR_PeakWE':'3minFTR', 'DA_week_PeakWE':'WeekDA', 'RT_week_PeakWE':'WeekRT'}
        
    df_distr_price.rename(columns=dict_rename_header,inplace=True)
 
    # remove "undefined" or "error" lines
    df_distr_price = df_distr_price[(df_distr_price.StDev.fillna("").astype(str)!="undefined") & (df_distr_price.StDev.fillna("").astype(str)!="error")]
    df_distr_price = df_distr_price[~df_distr_price.PredictedPrice.astype(str).str.contains("error")] 
    df_distr_price = df_distr_price[~df_distr_price.StDev.astype(str).str.contains("error")] 
    df_distr_price.PredictedPrice=pd.to_numeric(df_distr_price.PredictedPrice, errors = 'coerce', downcast = 'float').astype(np.float16).fillna(1000000)
    df_distr_price.StDev=pd.to_numeric(df_distr_price.StDev, errors = 'coerce', downcast = 'float').astype(np.float16).fillna(1000000)
    
    df_distr_price = df_distr_price[~df_distr_price['3minFTR'].astype(str).str.contains("NoData")] 
    
    df_distr_price = df_distr_price[~df_distr_price['StDevMinus'].astype(str).str.contains("error")] 
    df_distr_price = df_distr_price[~df_distr_price['StDevPlus'].astype(str).str.contains("error")] 
    df_distr_price = df_distr_price[~df_distr_price.WeekDA.astype(str).str.contains("error")]
    df_distr_price = df_distr_price[~df_distr_price.WeekRT.astype(str).str.contains("error")]
    
    df_distr_price['PredictedPrice'] = df_distr_price.PredictedPrice.astype(np.float16)
    df_distr_price['PredictedPriceOld'] = df_distr_price.PredictedPrice.astype(np.float16)
    df_distr_price['StDevPlus'] = df_distr_price.StDevPlus.astype(np.float16)
    df_distr_price['StDevMinus'] = df_distr_price.StDevMinus.astype(np.float16)
    df_distr_price['minDART'] = df_distr_price.minDART.astype(np.float16)
    df_distr_price['WeekDA'] = df_distr_price.WeekDA.astype(np.float16)
    df_distr_price['WeekRT'] = df_distr_price.WeekRT.astype(np.float16)
    
    
    df_points['SourceSink'] = df_points.Source+df_points.Sink
    
    # if down price - set as price max of (pr-1.5*sigma) or 3minFTR value
    if IS_DOWN_PRICE:
        
        df_distr_price['Pr_2sig'] = df_distr_price.PredictedPrice - (coef_sigma_down*df_distr_price.StDevMinus)
        df_distr_price['3minFTR'] = df_distr_price['3minFTR'].map(lambda x: {"NoData":-100000}.get(x,x)).astype(np.float32)
        if ignor_3minFTR_for_down_price:
            df_distr_price.PredictedPrice = df_distr_price['Pr_2sig']
        else:
            df_distr_price.PredictedPrice = df_distr_price[['3minFTR', 'Pr_2sig']].max(axis=1)
        df_distr_price.to_csv(get_common_root(log_path, 'Down_Price_'+os_path.basename(CLUSTERS_price_distribution_file)), index= False)
        df_distr_price.loc[df_distr_price.PredictedPrice != df_distr_price.Pr_2sig,'StDevMinus'] = (df_distr_price.StDevMinus/2)
        df_distr_price.loc[df_distr_price.StDevPlus > (df_distr_price.PredictedPriceOld- df_distr_price.PredictedPrice)/1.29 ,'StDevPlus'] = ((df_distr_price.PredictedPriceOld- df_distr_price.PredictedPrice)/1.29)
        
        dict_path_to_pred_exod = pd.Series(df_distr_price.PredictedPriceOld.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
        df_points['PredictionExodus'] = pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_to_pred_exod.get(x,1000000)).fillna(1000000)).round(3)
        
    # remove paths with not enough RT and DA sum if needed
    if USE_MIN_RT_DA:
        df_distr_price = df_distr_price[~df_distr_price.DA.astype(str).str.contains("error")]
        df_distr_price = df_distr_price[~df_distr_price.RT.astype(str).str.contains("error")]
        df_distr_price.DA = pd.to_numeric(df_distr_price.DA, errors = 'coerce', downcast = 'float').astype(np.float16).fillna(-1000000)
        df_distr_price.RT = pd.to_numeric(df_distr_price.RT, errors = 'coerce', downcast = 'float').astype(np.float16).fillna(-1000000)
        df_distr_price_low_RTDA = df_distr_price[(df_distr_price.RT < min_RT) | (df_distr_price.DA < min_DA)]
        list_low_RTDA_paths = (df_distr_price_low_RTDA.Source + df_distr_price_low_RTDA.Sink).tolist()
        # key = Source+Sink, value = DA sum
        dict_path_to_DA_sum = pd.Series(df_distr_price_low_RTDA.DA.values, index=(df_distr_price_low_RTDA.Source + df_distr_price_low_RTDA.Sink)).to_dict()
        # key = Source+Sink, value = RT sum
        dict_path_to_RT_sum = pd.Series(df_distr_price_low_RTDA.RT.values, index=(df_distr_price_low_RTDA.Source + df_distr_price_low_RTDA.Sink)).to_dict()
                
        df_points_low_RTDA = df_points[df_points.SourceSink.isin(list_low_RTDA_paths)]
        df_points_low_RTDA['RT'] = df_points_low_RTDA.SourceSink.map(dict_path_to_RT_sum).fillna('')
        df_points_low_RTDA['DA'] = df_points_low_RTDA.SourceSink.map(dict_path_to_DA_sum).fillna('')
        columns_for_output = ['Source','Sink','Source1','Sink1','MWs','Capital','flagStop','Comments','Comments_neg','Comments_coef','Take_constraints','Take_constraints_byOneNode','Fp_i','FV','Fp_i0','Price','Fp_i03','Price3','Collateral','iLoop']
        df_points_low_RTDA.to_csv(get_common_root(log_path,r'save2\0_RT_DA_sum_low.csv'), columns = columns_for_output, index= False)
        df_points = df_points[~df_points.SourceSink.isin(list_low_RTDA_paths)]
        del df_points_low_RTDA
    
    
    # since in price file could be only half of pathes (1-2 without 2-1) - restore other half (price = -price, sigma = sigma)
# =============================================================================
#     df_distr_price_back = df_distr_price.copy()
#     df_distr_price_back.rename(columns={'Source':'Sink', 'Sink':'Source'},inplace=True)
#     df_distr_price_back.PredictedPrice = - df_distr_price_back.PredictedPrice
#     
#     df_distr_price = pd.concat([df_distr_price, df_distr_price_back], ignore_index=True, sort=False)
#     df_distr_price.drop_duplicates(inplace=True)
# =============================================================================
    
    # key = Source+Sink, value = mean
    dict_path_to_mean = pd.Series(df_distr_price.PredictedPrice.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    dict_path_minDART = pd.Series(df_distr_price.minDART.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    dict_path_WeekDA = pd.Series(df_distr_price.WeekDA.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    dict_path_WeekRT = pd.Series(df_distr_price.WeekRT.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
# =============================================================================
#     # key = Source+Sink, value = sigma
#     dict_path_to_sigma = pd.Series(df_distr_price.StDev.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
# =============================================================================
    # key = Source+Sink, value = sigma+
    dict_path_to_sigma_plus = pd.Series(df_distr_price.StDevPlus.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    # key = Source+Sink, value = sigma-
    dict_path_to_sigma_minus = pd.Series(df_distr_price.StDevMinus.values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    # key = Source+Sink, value = 3minFTR
    dict_path_to_3minFTR = pd.Series(df_distr_price['3minFTR'].values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
     
    probability_coeff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sigma_coeff = [-1.29, -0.85, -0.53, -0.26, 0, 0.26, 0.53, 0.85, 1.29]
    
    #probability_coeff = [0.5, 0.6, 0.7, 0.8, 0.9]
    #sigma_coeff = [0, 0.26, 0.53, 0.85, 1.29]
    

    df_points['Mean_price'] = pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_to_mean.get(x,1000000)).fillna(1000000)).round(3)
    df_points['3minFTR']= pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_to_3minFTR.get(x,1000000)).fillna(1000000)).round(3)
    df_points['Sigma_minus']= pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_to_sigma_minus.get(x,1000000)).fillna(1000000)).round(3)
    df_points['Sigma_plus']= pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_to_sigma_plus.get(x,1000000)).fillna(1000000)).round(3)
    df_points['minDART']= pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_minDART.get(x,1000000)).fillna(1000000)).round(3)
    df_points['WeekDA']= pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_WeekDA.get(x,1000000)).fillna(1000000)).round(3)
    df_points['WeekRT']= pd.to_numeric(df_points.SourceSink.map(lambda x: dict_path_WeekRT.get(x,1000000)).fillna(1000000)).round(3)
# =============================================================================
#     df_points['Sigma_price'] = pd.to_numeric(df_points.SourceSink.map(dict_path_to_sigma).fillna(0)).round(3)                #######
#     df_points['Sigma_minus'] = np.where(df_points.Sigma_price > 200, 200, df_points.Sigma_price)                             #######
#     if not IS_DOWN_PRICE:
#         df_points['Sigma_minus'] =  np.minimum(df_points.Sigma_price, (df_points.Mean_price-df_points['3minFTR']).abs()/1.29)
#         df_points['Sigma_minus'] = np.where(df_points.Mean_price  < df_points['3minFTR'], 100, df_points.Sigma_minus)
#         
#     df_points['Sigma_plus'] = np.where(df_points.Mean_price.abs() >= 150, 
#              np.maximum(df_points.Sigma_price, 0.3*df_points.Mean_price.abs())
#              , 
#              np.maximum(df_points.Sigma_price, 45))
# =============================================================================
    print('get mean and sigma')
    
    #file_save_path_with_absent_price = os_path.join(log_path,'Path_with_absent_price_data.csv')
    df_points_bad = df_points[(df_points.Mean_price.abs() == 1000000) |(df_points['3minFTR'].abs() == 1000000)].copy()
    df_points = df_points[df_points.Mean_price.abs() != 1000000]
    
   
    
    for i_price in range(len(probability_coeff)):
        sigma_coef_i = sigma_coeff[i_price]
        if sigma_coef_i > 0:
            df_points['Price_'+str(i_price)] = df_points.Mean_price+df_points.Sigma_plus*sigma_coef_i
        else:
            df_points['Price_'+str(i_price)] = df_points.Mean_price+df_points.Sigma_minus*sigma_coef_i
        # since afterward we choose paths with FV-Price >= some_value, we should as F take (FV-Price-some_value )   
        df_points['F_'+str(i_price)] = (df_points.FV - df_points['Price_'+str(i_price)] - np.where(df_points['Price_'+str(i_price)].abs()*min_price_fraction_delta >= min_price_delta, df_points['Price_'+str(i_price)].abs()*min_price_fraction_delta, min_price_delta ))*probability_coeff[i_price]
        
        # for options do not take in account negative prices - so put  F = -9999 for that case - so if other prices are positive - choose them
        if IS_OPTION:
            df_points.loc[df_points['Price_'+str(i_price)] < 0, 'F_'+str(i_price)] = -9999
        ###
    for i_price3 in range(len(probability_coeff)):
        sigma_coef_i = sigma_coeff[i_price3]
        if sigma_coef_i > 0:
            df_points['Price3_'+str(i_price3)] = df_points.Mean_price+df_points.Sigma_plus*sigma_coef_i
        else:
            df_points['Price3_'+str(i_price3)] = df_points.Mean_price+df_points.Sigma_minus*sigma_coef_i
        # since afterward we choose paths with FV-Price >= some_value, we should as F take (FV-Price-some_value )   
        df_points['F3_'+str(i_price3)] = (df_points.FV - df_points['Price3_'+str(i_price3)] - np.where(df_points['Price3_'+str(i_price3)].abs()*(min_price_fraction_delta*2) >= (2*min_price_delta), df_points['Price3_'+str(i_price3)].abs()*(min_price_fraction_delta*2), (2*min_price_delta)))*probability_coeff[i_price3]
        
        # for options do not take in account negative prices - so put  F = -9999 for that case - so if other prices are positive - choose them
        if IS_OPTION:
            df_points.loc[df_points['Price3_'+str(i_price3)] < 0, 'F3_'+str(i_price3)] = -9999
        ###
    print('get prices and F-s')    
        
    f_columns =  ['F_'+str(i_price) for i_price in range(len(probability_coeff)) ]
    ###
    f3_columns =  ['F_'+str(i_price3) for i_price3 in range(len(probability_coeff)) ]
    ###
    dict_f_col_to_N = dict(zip(f_columns, range(len(probability_coeff))))
    df_points['max_F_ind'] = df_points[f_columns].idxmax(axis=1).map(dict_f_col_to_N)
    print('get max ind')
    df_points['max_F'] = df_points[f_columns].max(axis=1)
    for i_price in range(len(probability_coeff)):
        df_points.loc[df_points.max_F_ind == i_price, 'Price'] = df_points['Price_'+str(i_price)]
    # for options there are no negative prices - so set price to zero
    if IS_OPTION:
         df_points.loc[df_points.Price < 0, 'Price'] = 0
   
    df_points['Price_probability'] = df_points.max_F_ind.map(lambda x: probability_coeff[x])
    
    df_points_bad['add_to_FV'] = df_points_bad.FV.map(lambda x: max(min_price_fraction_delta*x,min_price_delta))
    df_points_bad['Price'] = df_points_bad.FV - df_points_bad.add_to_FV
    if IS_OPTION:
         df_points_bad.loc[df_points_bad.Price < 0, 'Price'] = 0
    df_points_bad['max_F'] = df_points_bad.FV - df_points_bad.Price
    
    ###
    dict_f3_col_to_N = dict(zip(f3_columns, range(len(probability_coeff))))
    df_points['max_F3_ind'] = df_points[f3_columns].idxmax(axis=1).map(dict_f3_col_to_N)
    print('get max ind')
    df_points['max_F3'] = df_points[f3_columns].max(axis=1)
    for i_price3 in range(len(probability_coeff)):
        df_points.loc[df_points.max_F3_ind == i_price3, 'Price3'] = df_points['Price3_'+str(i_price3)]
    # for options there are no negative prices - so set price to zero
    if IS_OPTION:
         df_points.loc[df_points.Price3 < 0, 'Price3'] = 0
   
    df_points['Price3_probability'] = df_points.max_F3_ind.map(lambda x: probability_coeff[x])
    
    df_points_bad['add_to_FV3'] = df_points_bad.FV.map(lambda x: max((2*min_price_fraction_delta)*x,(2*min_price_delta)))
    df_points_bad['Price3'] = df_points_bad.FV - df_points_bad.add_to_FV3
    if IS_OPTION:
         df_points_bad.loc[df_points_bad.Price3 < 0, 'Price3'] = 0
    df_points_bad['max_F3'] = df_points_bad.FV - df_points_bad.Price3
    ###
    
    # for bad points set max_F_ind  to  = -1 to throw it to 4th stage 
    df_points_bad['max_F_ind'] = -1
    # in predicted prices file there is error column = 3MonthFTR. if it true - path is good, and set 'max_F_ind' = -1 otherwise
    if '3MonthFTR' not in df_distr_price.columns:
        df_distr_price['3MonthFTR'] = True
    dict_path_to_error = pd.Series(df_distr_price['3MonthFTR'].values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    df_points['Er'] = df_points.SourceSink.map(lambda x: dict_path_to_error.get(x,False)).fillna(False)
    df_points.loc[df_points.Er==False, 'max_F_ind'] = -1
    
    ###df_points = pd.concat([df_points,df_points_bad], sort=False)
    
    del df_points['Er']
    
    ###
    df_points_bad['max_F3_ind'] = -1
    # in predicted prices file there is error column = 3MonthFTR. if it true - path is good, and set 'max_F_ind' = -1 otherwise
    if '3MonthFTR' not in df_distr_price.columns:
        df_distr_price['3MonthFTR'] = True
    dict_path_to_error3 = pd.Series(df_distr_price['3MonthFTR'].values, index=(df_distr_price.Source+df_distr_price.Sink)).to_dict()
    df_points['Er3'] = df_points.SourceSink.map(lambda x: dict_path_to_error3.get(x,False)).fillna(False)
    df_points.loc[df_points.Er3==False, 'max_F3_ind'] = -1
    
    df_points = pd.concat([df_points,df_points_bad], sort=False)
    ###
    del df_points['SourceSink']
    del df_points['Er3']
    
    return df_points

def calc_Price_from_node_price_file(df_points, df_shadow_price, IS_ANNUAL_PRICES = False, ANNUAL_PRICE_MODE = ''):
    # for normal monthly prices calculate price of path as Source - Sink 
    if not IS_ANNUAL_PRICES:
        # save to dict key = Point name, values = Price
       dict_points_price = pd.Series(df_shadow_price.ShadowPrice.values,index=df_shadow_price.SourceSink).to_dict()
        
       df_points['PriceSource'] = df_points.Source.map(dict_points_price).astype(np.float16)
       df_points['PriceSink'] = df_points.Sink.map(dict_points_price).astype(np.float16)
       # calc prices for paths Source - Sink 
       df_points.loc[(df_points.PriceSource.notnull()) & (df_points.PriceSink.notnull()),'Price'] = (df_points['PriceSource'] - df_points['PriceSink'])
       
       del df_points['PriceSource'];
       del df_points['PriceSink'];
       
    # for annual prices for each path set min, median and max prices from all periods
    if IS_ANNUAL_PRICES:
        periods_columns = df_shadow_price.columns[df_shadow_price.columns.str.match(".*_..")]
        
        dict_period_to_dict_nodePrice = dict()
        
        for period in periods_columns:
           dict_tmp = pd.Series(df_shadow_price[period].values,index=df_shadow_price.SourceSink).to_dict()
           dict_period_to_dict_nodePrice[period] = dict_tmp
    
        for period in periods_columns:
          df_points[period] = df_points.Source.map(dict_period_to_dict_nodePrice[period]) - df_points.Sink.map(dict_period_to_dict_nodePrice[period])

        df_points['Price_max'] = df_points[periods_columns].max(axis=1)
        df_points['Price_median'] = df_points[periods_columns].median(axis=1)
        df_points['Price_min'] = df_points[periods_columns].min(axis=1)
       
        df_points.loc[df_points['Price_median'].isnull(),'Price_median'] = df_points.Price_max[df_points['Price_median'].isnull()]
       
        dict_price_mode = {'MAX':'Price_max', 'MEDIAN':'Price_median', 'MIN':'Price_min'}
        df_points['Price'] = df_points[dict_price_mode[ANNUAL_PRICE_MODE]]
        
        for period in periods_columns:
            del df_points[period]
    
    df_points['max_F'] = df_points.FV - df_points.Price
        
    return df_points

def calc_Collateral(df_points, CLUSTERS_iso, IS_UNITARY_COLLATERAL, CLUSTER_collateral, min_collateral_value, class_peak):
    
    # Collateral column - used for F calculation, 
    # Collateral_t - used for cacl collateral capital - real collateral spent    
    
    if (IS_UNITARY_COLLATERAL == 1) | (IS_UNITARY_COLLATERAL == True):
        df_points['Collateral'] = 1
        df_points['Collateral_t'] = 1
        
        return df_points
    
    if CLUSTERS_iso == 'SPP':
        df_collateral = pd.read_csv(CLUSTER_collateral,sep='|')
        df_collateral['NC'] = df_collateral.SOURCE_LOCATION + df_collateral.SINK_LOCATION
        dict_collateral = pd.Series(df_collateral.PRODUCT_REFERENCE_PRICE.fillna(0).values,index=df_collateral.NC).to_dict()
        df_points['NC'] = df_points.Source+df_points.Sink
        df_points['NC_price'] = df_points.NC.map(dict_collateral).fillna(0)
        df_points['Collateral'] = df_points.Price - df_points.NC_price
        df_points.loc[df_points.Collateral < 0, 'Collateral'] = 0
        df_points['Collateral_t'] = df_points['Collateral']        
        df_points['Collateral_SPP'] = df_points['Collateral']
        
        del df_points['NC_price']
        del df_points['NC']
        
    elif CLUSTERS_iso == 'ERCOT':
        df_collateral = pd.read_csv(CLUSTER_collateral)
        df_collateral['NC'] = df_collateral.Source + df_collateral.Sink
        # if in First-Time-appear column there is comment started with "There is" - means wrong data in sum - replace it with 0
        df_collateral.loc[df_collateral['First-Time-appear'].str.contains("There is"),'Sum'] = 0
        dict_collateral = pd.Series(df_collateral.Sum.fillna(0).values,index=df_collateral.NC).to_dict()
        df_points['NC'] = df_points.Source+df_points.Sink
        df_points['Collateral'] = df_points.NC.map(dict_collateral).fillna(0)
        df_points['Collateral_t'] = df_points['Collateral'] 
        del df_points['NC']
    
    elif CLUSTERS_iso == 'ISONE':
        df_collateral = pd.read_csv(CLUSTER_collateral)
        df_collateral = df_collateral[df_collateral.ClassType == class_peak]
        df_collateral['NC'] = df_collateral["Source Location Name"] + df_collateral["Sink Location Name"]
        dict_collateral = pd.Series(df_collateral.SRFA.fillna(0).values,index=df_collateral.NC).to_dict()
        df_points['NC'] = df_points.Source+df_points.Sink
        df_points['Collateral'] = df_points.NC.map(dict_collateral).fillna(0)
        df_points['Collateral_t'] = df_points['Collateral'] 
        del df_points['NC']
        df_points.loc[df_points.Collateral<=min_collateral_value,'Collateral'] = min_collateral_value
        df_points['Collateral'] = df_points['Collateral'].astype(np.int32)
        return df_points
    
    else:
        df_points['Collateral'] = df_points.Price
        df_points['Collateral_t'] = df_points['Collateral']
        df_points.loc[df_points.Collateral < 0,'Collateral'] = df_points.Collateral.abs()
        
    # to calculated collateral add price (if price < 0 -> add 0)
    df_points.loc[df_points.Price > 0, 'Collateral'] = df_points.Collateral + df_points.Price
    df_points.loc[df_points.Collateral<=min_collateral_value,'Collateral'] = min_collateral_value
    df_points['Collateral'] = df_points['Collateral'].astype(np.int32)
    

    return df_points

# calc_F_main = 0 - calc with full list of constraints
# calc_F_main = 1 - calc with only main constraints
# calc_F_main = 3 - calc with min sum for full list of constraints           
def calc_FV(df_points, calc_F_mode, dict_coeff_constraints, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef):   
    
    list_constraints_for_calc_F = list_constr_for_FV
    if calc_F_mode == 1: # only for main constr
        list_constraints_for_calc_F = list_main_constraints
    
     # remove 5min and "clear" constraints from F calculation
    list_constraints_for_calc_F = [constr for constr in list_constraints_for_calc_F if ('5min' not in constr  and '_clear' not in constr)]

    
    df_points['FV'] = 0
    df_points['k_for_FV'] = 0
    for constr in list_constraints_for_calc_F:
        
        # get [N-,N+], [monthSum_Min, monthSum_Max], Possibility for constr
        list_limits_constraints = dict_coeff_constraints[constr]

        sum_min = np.int32(list_limits_constraints[1][0])
        sum_max = np.int32(list_limits_constraints[1][1])
            
        possibility = list_limits_constraints[2]
        influence_vect = df_points[constr].astype(np.float16)
        
        df_points['k_for_FV'] = np.where(influence_vect > 0,CLUSTER_F_correction_coef, 1 )
        if calc_F_mode == 3:
            df_points['sum_for_FV'] = sum_min
        else:
            df_points['sum_for_FV'] = np.where(influence_vect > 0, sum_min, sum_max )
        # FV =  sum constr ( inf*sum*possib )
        df_points.FV += influence_vect*df_points.sum_for_FV*possibility   
        
    df_points.FV = df_points.FV.astype(np.int32)
    del df_points['k_for_FV']
    del df_points['sum_for_FV']
        
    return df_points


def calc_weight_path(df_points, dict_coeff_constraints, list_constr_for_FV, list_main_constraints, CLUSTER_F_weight_coef):   
    
    list_constraints_for_calc_weight = list_constr_for_FV
    
     # remove 5min and "clear" constraints from F calculation
    list_constraints_for_calc_weight = [constr for constr in list_constraints_for_calc_weight if ('5min' not in constr  and '_clear' not in constr)]
    
    df_points['Weight'] = 0
    df_points['k_for_Weight'] = 0
    df_points.Weight = df_points.Weight.astype(np.float16)
    df_points.k_for_Weight = df_points.k_for_Weight.astype(np.float16)
    for constr in list_constraints_for_calc_weight:
        
        # get [N-,N+], [monthSum_Min, monthSum_Max], Possibility for constr
        list_limits_constraints = dict_coeff_constraints[constr]
        
        coef_min = abs(np.float16(list_limits_constraints[0][0]))
        
        if coef_min == 0:
            coef_min = 0.00001
        
        influence_vect_weight = df_points[constr].astype(np.float16)
        
        df_points['k_for_Weight'] = np.where(influence_vect_weight < 0,CLUSTER_F_weight_coef, 0 )
        
        #df_points['coef_for_Weight'] = np.where(influence_vect > 0, sum_min, sum_max )
        #FV =  sum constr ( inf*sum*possib )
        df_points.Weight += (influence_vect_weight.astype(np.float16)*df_points.k_for_Weight.astype(np.float16)*-1)/coef_min
        
    df_points.Weight = df_points.Weight.astype(np.float16)
    
    del df_points['k_for_Weight']
    
    return df_points


# calc F of path itself
def calc_F_path0(df_points):
    
    df_points.Fp_i0 = (df_points.FV-df_points.Price)/df_points.Collateral
    
    ###
    df_points.Fp_i03 = (df_points.FV-df_points.Price3)/df_points.Collateral
    ###
    
    return df_points

# calc F of path taking in account current portfolio (current MW of given path)
# NOT IN USE NOW
def calc_F(df_points, MWstep):
    df_points['Fp_i'] = 0
    df_points.Fp_i = (MWstep*df_points.Fp_i0 + (df_points.MWs*df_points.Fp_i0).sum())/MWstep
    
    return df_points

# returns bool vector for every path:
# false if any constraint from list_constraints_for_calc_F  
# after adding given path excesses N+ and true otherwise
def fix_F_for_Nplus_excess(df_points, dict_coeff_constraints_orig_table, dict_coef_portfolio, MWstep, calc_F_main, list_constr_for_FV, list_main_constraints):
    
    if 'Fp_i' in df_points.columns:
        df_points.Fp_i = df_points.Fp_i0
    else:
        df_points['Fp_i'] = df_points.Fp_i0
    list_constraints_for_calc_F = list_constr_for_FV
    if calc_F_main == 1: # only for main constr
        list_constraints_for_calc_F = list_main_constraints
    
     # remove 5min constraints from F calculation
    list_constraints_for_calc_F = [constr for constr in list_constraints_for_calc_F if '5min' not in constr ]
    
    for constr in list_constraints_for_calc_F:
        # get [N-,N+], [monthSum_Min, monthSum_Max], Possibility for constr
        list_limits_constraints = dict_coeff_constraints_orig_table[constr]
        coefNplus = list_limits_constraints[0][1]
        sum_min = np.int32(list_limits_constraints[1][0])
        possibility = list_limits_constraints[2]
        influence_vect = df_points[constr].astype(np.float16)
        coef_portfolio = dict_coef_portfolio[constr]
        # if path takes constraint (influence > 0) with coeff > N+ (coef_portfolio > coefNplus),
        # means N+ exceed - make var Fpi less (remove contribution of this constraint)
        #nplus_excess_condition = (influence_vect > 0) & (coef_portfolio > coefNplus)
        # remove from Fpi contribution of constraint which exceeds N+ limit
        
        df_points.Fp_i = df_points.Fp_i.mask(((influence_vect > 0) & (coef_portfolio > coefNplus)), (df_points.Fp_i - ((influence_vect*sum_min*possibility)/df_points.Collateral)))
        
        #df_points.loc[nplus_excess_condition, 'Fp_i'] = df_points.Fp_i - ((influence_vect*sum_min*possibility)/df_points.Collateral)
        
        df_points.Fp_i = df_points.Fp_i.astype(np.float32)
    
    return df_points

def choose_cur_main_constr(df_points, list_main_constraints, untaken_main_constr, dict_constr_FV, IS_CHOOSE_CONSTR_BY_FV):
    
    # if all constraints are taken - just give the first one
    if len(untaken_main_constr) == 0:
        return list_main_constraints[0]
    
    dict_Npaths_to_constraint = dict()
    dict_constr_to_Npaths = dict()
    dict_FV_to_constr = dict()
      
    n_paths_all = list()
    all_FV = []
    for constr in list_main_constraints:
        n_paths = df_points[(df_points.Take_constraints.str.contains(constr, regex=False, na = False))].shape[0]            
        if constr in untaken_main_constr:
            n_paths_all.append(n_paths)                
            dict_Npaths_to_constraint[n_paths] = constr
            FV = dict_constr_FV[constr]
            all_FV.append(FV)
            dict_FV_to_constr[FV] = constr
        dict_constr_to_Npaths[constr] = n_paths
        
    # another way of choosing constraint  - by its FV = sum*possib
    if IS_CHOOSE_CONSTR_BY_FV:
        max_FV = sorted(all_FV)[-1]
        curr_constr_main = dict_FV_to_constr[max_FV]
    else: 
        # default way of choosing constr to analyse - by min amount of paths
        min_npath_key = sorted(n_paths_all)[0]
        curr_constr_main = dict_Npaths_to_constraint[min_npath_key]
    
    return curr_constr_main
            
def get_corr_clusters(iso,bid_class, CLUSTERS_RCTool, min_corr_value, df_points_sourcesink_source, log_path):
    
    #----------- part of RC tool correlations -----------------
    print('getting RC tool data')
    df_points_corr = get_RC_tool_data(CLUSTERS_RCTool)
     # set corr to 1, if value >=min_corr_value, not used
    df_points_corr.loc[(df_points_corr.corDA50>=min_corr_value) & (df_points_corr.corDA100>=min_corr_value) & (df_points_corr.corDA150>=min_corr_value), 'CorrDA'] = 1
    df_points_corr.loc[(df_points_corr.corRT50>=min_corr_value) & (df_points_corr.corRT100>=min_corr_value) & (df_points_corr.corRT150>=min_corr_value), 'CorrRT'] = 1
    df_points_corr.loc[(df_points_corr.corDA50>=min_corr_value) & (df_points_corr.corDA100>=min_corr_value) & (df_points_corr.corDA150>=min_corr_value) & (df_points_corr.corRT50>=min_corr_value) & (df_points_corr.corRT100>=min_corr_value) & (df_points_corr.corRT150>=min_corr_value), 'CorrALL'] = 1
    
    # check min value and count of corr values >0.995
    df_points_corr['CorrFinalMin']   = df_points_corr[['corDA50','corDA100','corDA150','corRT50','corRT100','corRT150']].where(lambda x: x>=min_corr_value, axis=1).min(axis=1)
    df_points_corr['CorrFinalCount'] = df_points_corr[['corDA50','corDA100','corDA150','corRT50','corRT100','corRT150']].where(lambda x: x>=min_corr_value, axis=1).count(axis=1)
    
    # points are same if min cor values > 0.99 in all corr periods (50 days ago, 100 days ago, 150 days ago in DA and RT)
    file_save_rc_tool_data = os_path.join(log_path,'Data_RC_Tool.csv')
    df_points_corr.to_csv(file_save_rc_tool_data,index=0)
    
    df_points_corr = df_points_corr[(df_points_corr.CorrFinalCount>=6)]
    
    # choose only those pairs which are in SourceSink file 
    df_points_corr = df_points_corr[(df_points_corr.Point1.isin(df_points_sourcesink_source.Name)) & (df_points_corr.Point2.isin(df_points_sourcesink_source.Name))]
    
    df_cluster1 = pd.DataFrame()
    df_cluster1['N1'] = df_points_corr['Point1']
    df_cluster1['N2'] = df_points_corr['Point2']
    
     #----------- part of same buses correlations -----------------
    # check points on same BUSes, concat all buses to one string and remove duplicates
    print('getting bus corr data')
    if 'BusName' in df_points_sourcesink_source.columns:
        df_points_sourcesink_group = df_points_sourcesink_source.groupby('Name').agg({'BusName':lambda x: ';'.join(set(x.sort_values()))}).reset_index()
        df_points_sourcesink_group['Mapped'] = df_points_sourcesink_group.groupby('BusName')['Name'].transform('first')
        
        file_save_all_paths = os_path.join(log_path,'Common_Bus_Source_Sink_from_SourceSink.csv')
        df_points_sourcesink_group.to_csv(file_save_all_paths, index=0)
    
        df_cluster2 = pd.DataFrame()
        df_cluster2['N1'] = df_points_sourcesink_group['Name']
        df_cluster2['N2'] = df_points_sourcesink_group['Mapped']
    
        df_cluster = pd.concat([df_cluster1,df_cluster2])
        
    else:
        df_cluster = df_cluster1
    
    df_cluster.drop_duplicates(inplace=True)
    
    print('getting dict rename')
    dict_rename = get_dict_rename(iso,bid_class, df_cluster,log_path)
    
    list_points_unique = list(set(df_points_sourcesink_source.Name.map(lambda x: dict_rename.get(x,x))))
    
    with open(os_path.join(log_path,'List_of_unique_points.txt'),'w') as f:
        for point in list_points_unique:
            f.write('{}\n'.format(point))
    
    return (list_points_unique, dict_rename)

def get_dict_rename(iso, bid_class, df_cluster,log_path):
    
    # create reverse pairs
    df_cluster_rev = df_cluster.rename(columns={'N1':'N2', 'N2':'N1'})
   
    all_nodes = list(set(pd.concat([df_cluster,df_cluster_rev],sort=True).N1.tolist()))
    
    # list of corr pairs [n1,n2]
    pairs = pd.concat([df_cluster,df_cluster_rev],sort=True).values.tolist()
    n_to_allN_dict = defaultdict(list)
    {n_to_allN_dict[key].append(value) for (key, value) in pairs}
    
    clusters = []
    rest_nodes = all_nodes
    while len(rest_nodes)>0:
        # get cluster for fist node in rest_nodes
        new_cluster = get_cluster([rest_nodes[0]], n_to_allN_dict)
        clusters.append(new_cluster)
        # remove nodes wich are already in cluster from list of nodes for which need to find cluster
        rest_nodes = [node for node in rest_nodes if node not in new_cluster]
    
    # sort clusters so the first node - is the node with longest history
    dict_node_to_live_days = get_dict_node_to_live_days(iso, bid_class)
    clusters = [ sorted(cluster, key=lambda x: dict_node_to_live_days.get(x,timedelta()),reverse=True)  for cluster in clusters]
    
    # write all clusters in file in format "main_node: [all cluster's nodes]"
    with open(os_path.join(log_path,'List_of_clusters.txt'),'w') as f:
        for cluster in clusters:
            f.write('{}: {}\n'.format(cluster[0],cluster))
    
    # for main point in cluster use the first one
    dict_rename = {node:cluster[0]  for cluster in clusters for node in cluster}
    return dict_rename
    
def get_cluster(cur_cluster, n_to_allN_dict):
    # get nodes related to all nodes in cur_cluster
    next_cluster = sorted(list(set([node for node_list in [n_to_allN_dict[node] for node in cur_cluster] for node in node_list])))
    if next_cluster==sorted(list(set(cur_cluster))):
        return next_cluster
    # do again untill list of next_clustr will not change
    return get_cluster(next_cluster+cur_cluster, n_to_allN_dict)

def get_dict_node_to_live_days(iso, bid_class):
    node_hist_file = glob.glob(os_path.join(get_path_to_node_history_lib(iso),'*'+bid_class.replace(' ', '')+'.csv'))[0]
    df_node_hist = pd.read_csv(node_hist_file)
    df_node_hist['start date'] = pd.to_datetime(df_node_hist['start date'], format='%Y-%m-%d')
    df_node_hist['end date'] = pd.to_datetime(df_node_hist['end date'], format='%Y-%m-%d')
    df_node_hist = df_node_hist.groupby('Pnode').agg({'start date':'min','end date':'max'}).reset_index()
    df_node_hist['live_days'] = df_node_hist['end date'] - df_node_hist['start date']
    dict_node_to_live_days = pd.Series(df_node_hist.live_days.values,index=df_node_hist.Pnode).to_dict()
    return dict_node_to_live_days
    
def finalize_loop(df_points, mws_max_path, constraints_taken_by_limits, n_main_constr, cur_Nplus_coef, n_plus_step,  max_N_plus, stage, calc_F_main, df_points_take_constraints_by_one_node,df_points_no_take_constraints, df_points_bad_price, dict_coeff_constraints, dict_coeff_constraints_orig_table, dict_coef_portfolio, dict_coef_portfolio_prev, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT):
    
    cond_good_df_points =  (df_points.flagStop==0) & (df_points.MWs<mws_max_path) 
    
    # if there is no paths for stage R- move to stage 3
    if (not cond_good_df_points.any()) & (stage == "R"):
        (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
    
    
    # if there is no paths for stage 3- move to stage 4
    if (not cond_good_df_points.any()) & (stage == 3):
        (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
    
    # if trying to take main constr
    if ((stage==1)|(stage==2)):
        
        take_constr_field = 'Take_constraints' if stage == 1 else 'Take_constraints_byOneNode'
        cond_good_df_points_take_constr = cond_good_df_points & (df_points[take_constr_field] != '')    
        
        # cond_normal_step - when no need to go to next stage or increase N+
        cond_normal_step = (cond_good_df_points_take_constr.any())  and (len(constraints_taken_by_limits) < n_main_constr)
        
        # if there are no paths which takes constraints or all main_constr were passed - reset vars and increase N+ or move to next stage
        if not cond_normal_step:
            
            constraints_taken_by_limits = set()
            df_points.loc[df_points.flagStop == -4, 'flagStop'] = 0            
            cur_Nplus_coef = cur_Nplus_coef + n_plus_step
            cond_good_df_points =  (df_points.flagStop==0) & (df_points.MWs<mws_max_path) 
            
            # cur_Nplus_coef > max_N_plus - move to next stage, else - increase N+
            if cur_Nplus_coef > max_N_plus or (not cond_good_df_points.any()) & (stage != 4) :
                
                LogStageValue('Loop {}: limit N+ {} > {} or no good points(cond_good_df_points.any() = {}) '.format(iLoop,cur_Nplus_coef,max_N_plus, cond_good_df_points.any()), CLUSTER_PATH_ROOT, "id")        
                # move to next stage
                (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
                # if df_points is still empty - move forvard to the next stage
                if df_points.shape[0] == 0:
                    (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
                    if df_points.shape[0] == 0:
                        (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
                        if df_points.shape[0] == 0:
                            (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
            else: # increase N+ (or move to next stage if portfolio coeff does not change )
    
                LogStageValue('Loop {}: increment limit N+ ({}) to {}'.format(iLoop,cur_Nplus_coef-n_plus_step,cur_Nplus_coef), CLUSTER_PATH_ROOT, "id")
                # set new N+ in dict
                for constr in list_main_constraints:
                    dict_coeff_constraints[constr][0][1] = cur_Nplus_coef
                    orig_Nplus_coef = dict_coeff_constraints_orig_table[constr][0][1]
                    if cur_Nplus_coef > orig_Nplus_coef:
                        dict_coeff_constraints[constr][0][1] = orig_Nplus_coef
                    # if previous portfolio coef was the same as current - add constraint to already taken list
                    # and if cur coef not more than current N+ (as could be on stage 2)
                    prev_coef_portf = dict_coef_portfolio_prev[constr]
                    cur_coef_portf = dict_coef_portfolio[constr]
                    if (prev_coef_portf == cur_coef_portf) & (cur_coef_portf < cur_Nplus_coef-n_plus_step) :
                        constraints_taken_by_limits.add(constr)
                        LogStageValue('Loop {}: no increment in N+ for constr {}, move it to taken list'.format(iLoop,constr), CLUSTER_PATH_ROOT, "id")
                        
                dict_coef_portfolio_prev = copy_deepcopy(dict_coef_portfolio)
                
                # if portfolio coeffs does not change for all of main constraints - move to next stage
                if len(set(constraints_taken_by_limits)) == len(set(list_main_constraints)):
                    (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints) = move_to_next_stage( stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT)
                    
                    dict_coef_portfolio_prev = defaultdict(int)
                    constraints_taken_by_limits = set()
            
    return (df_points, dict_coef_portfolio_prev, cond_good_df_points, constraints_taken_by_limits, stage, calc_F_main, cur_Nplus_coef, dict_coeff_constraints)
    
    
def move_to_next_stage(stage,calc_F_main,df_points,df_points_take_constraints_by_one_node,df_points_no_take_constraints,df_points_bad_price, cur_Nplus_coef, dict_coeff_constraints_orig_table, dict_coeff_constraints,dict_coef_portfolio,list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, n_plus_step, cond_good_df_points, iLoop, min_price_delta, min_price_fraction_delta, CLUSTER_PATH_ROOT):
    if stage == 1:
        stage = 2
        LogStageValue('Loop {}. Stage 2 begins'.format(iLoop), CLUSTER_PATH_ROOT, "id")
        df_points = pd.concat([df_points,df_points_take_constraints_by_one_node], ignore_index=True, sort = False)
        df_points.sort_values(by=['MWs'], ascending = True, inplace = True)
        df_points.drop_duplicates(['Source1','Sink1'], keep='last', inplace=True)
    elif stage == 2:
        stage = "R"      
        calc_F_main = 0
        list_constr_for_FV = sorted(list(dict_coeff_constraints.keys()))
        LogStageValue('Loop {}:  Stage R begins'.format(iLoop), CLUSTER_PATH_ROOT, "id")
        df_points_no_take_constraints = calc_FV(df_points_no_take_constraints, calc_F_main, dict_coeff_constraints,list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef)
        df_points_no_take_constraints = calc_F_path0(df_points_no_take_constraints)
        df_points = pd.concat([df_points,df_points_no_take_constraints], ignore_index=True, sort = False)
        df_points.sort_values(by=['MWs'], ascending = True, inplace = True)
        df_points.drop_duplicates(['Source1','Sink1'], keep='last', inplace=True)
        
    elif stage == "R":
        stage = 3      
        calc_F_main = 0
        LogStageValue('Loop {}:  Stage 3 begins'.format(iLoop), CLUSTER_PATH_ROOT, "id")
        # throw out paths which are not meet the conditions :
        #   for price <= 200  FV - price > 100
        #   for price > 200   FV - price > 1/2 price
        ###low_price_goodF_condition = (min_price_fraction_delta*df_points.Price <= min_price_delta) & ((df_points.FV - df_points.Price) > min_price_delta )
        ###high_price_goodF_condition = (min_price_fraction_delta*df_points.Price > min_price_delta) & ((df_points.FV - df_points.Price) > min_price_fraction_delta*df_points.Price )
        
        ###
        low_price_goodF_condition = ((2*min_price_fraction_delta)*df_points.Price3 <= (2*min_price_delta)) & ((df_points.FV - df_points.Price3) >= (2*min_price_delta) )
        high_price_goodF_condition = ((2*min_price_fraction_delta)*df_points.Price3 > (2*min_price_delta)) & ((df_points.FV - df_points.Price3) > ((2*min_price_fraction_delta)*df_points.Price3) )
        ###
        # set flagStop = 6 for negative F
        df_points.loc[~(low_price_goodF_condition | high_price_goodF_condition) & (df_points.MWs==0),'flagStop'] = 6     
        ###
        #set flagStop = 0 for all paths from stage R
        df_points.loc[df_points.flagStop == -10, 'flagStop'] = 0
        
        df_points.loc[df_points.MWs == 0, 'Price'] = df_points['Price3']
        df_points.loc[df_points.MWs == 0, 'Fp_i0'] = df_points['Fp_i03']
        ###
        #root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df100_{}.csv'.format(iLoop))
        #df_points.to_csv(root_array_df)
    else:
        stage = 4
        LogStageValue('Loop {}. Stage 4 begins'.format(iLoop), CLUSTER_PATH_ROOT, "id")
        df_points = pd.concat([df_points, df_points_bad_price], ignore_index=True, sort = False)
        df_points.sort_values(by=['MWs'], ascending = True, inplace = True)
        df_points.drop_duplicates(['Source1','Sink1'], keep='last', inplace=True)
        #root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df200_{}.csv'.format(iLoop))
        #df_points.to_csv(root_array_df)
        
    min_portfolio_Nplus_coef = min([dict_coef_portfolio[k] for k in list_main_constraints])+n_plus_step
    for constr in list_main_constraints:
        dict_coeff_constraints[constr][0][1] = min(min_portfolio_Nplus_coef, dict_coeff_constraints_orig_table[constr][0][1])
        
    cur_Nplus_coef = min_portfolio_Nplus_coef
    cond_good_df_points =  (df_points.flagStop==0)
    cond_good_df_points.values[:] = True
    if 'Fp_i' in df_points.columns:
        df_points.Fp_i = 0
    else:
        df_points['Fp_i'] = 0
    df_points['Comments_coef'] = ''
    df_points.loc[df_points.flagStop == -4, 'flagStop'] = 0
    df_points.loc[df_points.flagStop == -5, 'flagStop'] = 0
    
    return (df_points, stage, calc_F_main, cur_Nplus_coef, cond_good_df_points, dict_coeff_constraints)