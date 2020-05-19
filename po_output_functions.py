# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:36:34 2019

@author: Администратор
"""

from os import path as os_path, listdir as os_listdir, makedirs as os_makedirs
import pandas as pd
import numpy as np
import sys

folderSnippets = r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\snippets'
sys.path.insert(0,folderSnippets)
from fileIO import ReadOnly, get_dashboard
sys.path.insert(0, r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\pathsOptimizer")
from po_data_load_functions import get_constr_table_data

def write_portfolio_file(output_path, iLoop, df_portfolio_final, list_column_order):
    file_portfolio = os_path.join(output_path,'portfolio_{}.csv'.format(iLoop))
    df_portfolio_final.sort_values('Step').to_csv(file_portfolio,index=False, columns=list_column_order, float_format='%.3f')
    ReadOnly(file_portfolio)
    
def write_constraints_file(output_path, iLoop, df_portfolio_final, list_All_constraints, list_All_mws_constraints, dict_constr_in_dipoles, dict_coef_portfolio, dict_coef_portfolio_neg, dict_main_constraints, dict_coeff_constraints):
    
    file_constraints = os_path.join(output_path,'constraints_{}.csv'.format(iLoop)) 

    df_constraints = pd.DataFrame()    

    df_constraints['conID'] = list_All_constraints    
    df_constraints['Dipoles'] = df_constraints.conID.map(dict_constr_in_dipoles)
    df_constraints['Zone'] = ''
    
    df_constraints['Avg Coef'] = df_constraints.conID.map(dict_coef_portfolio).round(2) 
    df_constraints['Avg Neg. Coef'] = df_constraints.conID.map(dict_coef_portfolio_neg).round(2)  
    
    df_constraints['Capital'] = ''
    df_constraints['Coeff Peak'] = ''
    df_constraints['Coeff Off-Peak'] = ''
    
    dict_mw_constr_to_neg_paths = dict()
    for mw_constr in list_All_mws_constraints:
        if mw_constr in df_portfolio_final.columns:
            portfolio_neg = df_portfolio_final[df_portfolio_final[mw_constr]<0].copy()
        else: continue
        #negpaths in  form 'source-sink-[coef]'
        dict_mw_constr_to_neg_paths[mw_constr] = ';'.join((portfolio_neg.Source+'-'+portfolio_neg.Sink+'-['+portfolio_neg[mw_constr].round(2).astype(str)+']').tolist())
        
    df_constraints['mw_constr'] = list_All_mws_constraints
    df_constraints['NegativePaths'] = df_constraints.mw_constr.map(dict_mw_constr_to_neg_paths)
    df_constraints['PathZone'] = ''
    df_constraints['Driver'] = ''
    df_constraints['Comment'] = df_constraints.conID.map(dict_main_constraints)
    df_constraints['Season/MainComment'] = ''
    df_constraints['N-'] = df_constraints.conID.map(lambda x: dict_coeff_constraints[x][0][0])
    df_constraints['N+'] = df_constraints.conID.map(lambda x: dict_coeff_constraints[x][0][1])
    df_constraints['SumConstr'] = df_constraints.conID.map(lambda x: dict_coeff_constraints[x][1][0])
    df_constraints['Possibility'] = df_constraints.conID.map(lambda x: dict_coeff_constraints[x][2])
    df_constraints['Available_paths'] = ''
    
    df_constraints.sort_values('Avg Coef', ascending=False).to_csv(file_constraints, index=False,columns='conID,Dipoles,Zone,Avg Coef,Avg Neg. Coef,Capital,Coeff Peak,Coeff Off-Peak,MW Pos Peak,MW Neg Peak,MW Pos Off-Peak,MW Neg Off-Peak,NegativePaths,PathZone,Driver,Comment,Season/MainComment,N-,N+,SumConstr,Possibility,Constr_stop,Available_paths'.split(','))
    
    file_main_constr = file_constraints.replace('.csv','_main.csv')
    df_constraints[df_constraints.Comment.fillna('').str.contains('MainConstr')].sort_values('Avg Coef', ascending=False).to_csv(file_main_constr, index=False,columns='conID,Dipoles,Zone,Avg Coef,Avg Neg. Coef,Capital,Coeff Peak,Coeff Off-Peak,MW Pos Peak,MW Neg Peak,MW Pos Off-Peak,MW Neg Off-Peak,NegativePaths,PathZone,Driver,Comment,Season/MainComment,N-,N+,SumConstr,Possibility,Constr_stop,Available_paths'.split(','))
    
    ReadOnly(file_constraints)
    ReadOnly(file_main_constr)

def create_final_not_taken_paths_files(iso, output_path, not_taken_paths_folder, df_points_non_zero_infl, constr_list, MWs_step, dict_coeff_constraints_orig_table):

    not_taken_paths_filelist = os_listdir(not_taken_paths_folder) 
    not_taken_paths_filelist = list(map(lambda f: os_path.join(not_taken_paths_folder,f),not_taken_paths_filelist)) # add folder path - make full path
    df_not_taken_paths = [pd.read_csv(file) for file in not_taken_paths_filelist]
    df_not_taken_paths = pd.concat(df_not_taken_paths)
    # create rows in format [constrName, Step, Npaths] 
    #out_columns_blocking_constr = ['Constraint','Step','Npaths','N-', 'N+']
    out_columns_blocking_constr = ['Constraint', 'ConstraintDevice', 'ContingencyDevice', 'Step', 'Npaths', 'AllNpaths', 'N-', 'N+', 'FV', 'Main', 'OldStatus', 'DisappearStatus', 'Seasonality']
    
    
    # set dict key for dict_coeff_constraints_orig_table = Constraint+Contingency+Direction and value = list [[N-,N+],[sumMin, sumMax],Possibility,Main,OldStatus,DisappearStatus,Seasonality,ConstraintDevice,ContingencyDevice,FV]
    rows = []
    for constr in constr_list:
        data_of_cur_constr = df_not_taken_paths[df_not_taken_paths.Comments_neg.fillna('').str.contains(constr, regex=False)]
        
        data_grouped_by_loop =  data_of_cur_constr.groupby('iLoop')
        n_loops = len(data_grouped_by_loop) # amount of steps when cur constraint block paths
        loops = list(data_grouped_by_loop.groups.keys()) #  steps numbers when cur constraint block paths
        Npaths = list(data_grouped_by_loop.size()) #  numbers of paths blocked by cur constr
        N_minus = dict_coeff_constraints_orig_table[constr][0][0]
        N_plus = dict_coeff_constraints_orig_table[constr][0][1]
        AllNpaths = 0 # sum Npaths for one ConstraintDevice (not conID)
        FVconst = dict_coeff_constraints_orig_table[constr][9]
        mainConst = dict_coeff_constraints_orig_table[constr][3]
        oldStat = dict_coeff_constraints_orig_table[constr][4]
        disapStat = dict_coeff_constraints_orig_table[constr][5]
        seasonConst = dict_coeff_constraints_orig_table[constr][6]
        deviceConst = dict_coeff_constraints_orig_table[constr][7]
        contingConst = dict_coeff_constraints_orig_table[constr][8]
        
        
        for i in range(n_loops):
            #rows.append([constr, loops[i], Npaths[i], N_minus, N_plus] )
            rows.append([constr, deviceConst, contingConst, loops[i], Npaths[i], AllNpaths, N_minus, N_plus, FVconst, mainConst, oldStat, disapStat, seasonConst] )
    
    # write to output file       
    df_step_n_paths = pd.DataFrame(rows, columns = out_columns_blocking_constr)
    
    list_constrDEV = list(df_step_n_paths['ConstraintDevice'])
    for constrDEV in list_constrDEV:
        df_step_n_paths.loc[df_step_n_paths.ConstraintDevice == constrDEV,'AllNpaths'] = sum(df_step_n_paths.Npaths[df_step_n_paths.ConstraintDevice == constrDEV])

    output_folder_not_taken_paths = os_path.join(output_path,'not_taken_paths')
    if not os_path.exists(output_folder_not_taken_paths):
        os_makedirs(output_folder_not_taken_paths)
    output_file = os_path.join(output_folder_not_taken_paths, iso+'_blocking_constraints.csv')
    df_step_n_paths.sort_values('Step').to_csv(output_file, index=False,columns=out_columns_blocking_constr, float_format='%.3f')
    
    # full file with ['Source', 'Sink', 'ConID', 'ConstraintDevice', 'ContingencyDevices', 'Direction', 'Influence', 'Step', 'Reason'] columns
    #columns_full_file = ['Source', 'Sink', 'ConID', 'ConstraintDevice', 'ContingencyDevices', 'Direction', 'Influence', 'Step', 'Reason']
    columns_full_file = ['Source', 'Sink',  'Constraint',  'Influence', 'Step', 'Reason', 'flagStop','Fp_i','FV','Fp_i0','Price','Fp_i03','Price3','Collateral','Collateral_t','Price_0','Price3_0', 'iLoop', 'Weight']
    df_not_taken_paths['Reason'] = ''
    dict_flagStop_to_reason = {3:'blocked by RM: ', 4:'max MW node', 1:'max MW path', 10:'negative F', 6:'negative F stage3',7:'neg delta F', 5:'neg recentDART', 0:'0_o', 15:'hack_FV_BOT', 16:'hack_Price_TOP', 17:'hack_Price_BOT', 18:'hack_Collat_TOP', 19:'hack_Collat_BOT'}
    df_not_taken_paths.Reason =  df_not_taken_paths.flagStop.map(dict_flagStop_to_reason)
    df_not_taken_paths.loc[df_not_taken_paths.flagStop == 3,'Reason'] = df_not_taken_paths.loc[df_not_taken_paths.flagStop == 3,'Reason'] + df_not_taken_paths.loc[df_not_taken_paths.flagStop == 3,'Comments_neg']
    #df_not_taken_paths.loc[df_not_taken_paths.flagStop == 3,'Reason'] = df_not_taken_paths.Reason[df_not_taken_paths.flagStop == 3] + df_not_taken_paths.Comments_neg[df_not_taken_paths.flagStop == 3]
    
    # not taken statistics
    pd_stat = df_not_taken_paths.groupby('flagStop').size().reset_index(name='Npaths')
    pd_stat['Reason'] = ''
    pd_stat.Reason =  pd_stat.flagStop.map(dict_flagStop_to_reason)
    output_file = os_path.join(output_folder_not_taken_paths, iso+'_not_taken_paths_statistics.csv')
    pd_stat.sort_values('Npaths').to_csv(output_file, index=False, float_format='%.3f')

    
    # split data to  data with flagStop == 3 (stopped by constraints) and all other
    df_not_taken_paths_constr_driven_reason = df_not_taken_paths[df_not_taken_paths.flagStop == 3]
    #df_not_taken_paths_constr_driven_reason = df_not_taken_paths[df_not_taken_paths.flagStop == 3]
    df_not_taken_paths_not_constr = pd.concat([df_not_taken_paths, df_not_taken_paths_constr_driven_reason, df_not_taken_paths_constr_driven_reason]).drop_duplicates(keep=False)
    
    # write non constraints data to file
    df_not_taken_paths_not_constr.rename(columns={'iLoop': 'Step'}, inplace=True)
    df_not_taken_paths_not_constr['Influence'] = df_not_taken_paths_not_constr.Fp_i*df_not_taken_paths_not_constr.Collateral*MWs_step
    output_file = os_path.join(output_folder_not_taken_paths, iso+'_not_taken_paths_no_constraints.csv')
    df_not_taken_paths_not_constr.sort_values('Step').to_csv(output_file, index=False,columns=columns_full_file, float_format='%.3f')
    
    # choose only needed columns
    short_cols = ['Source', 'Sink', 'iLoop', 'Reason','Fp_i','FV','Fp_i0','Price','Fp_i03','Price3','Collateral','Collateral_t','Price_0','Price3_0', 'flagStop', 'Weight']
# =============================================================================
#     df_not_taken_paths_short = df_not_taken_paths_constr_driven_reason[short_cols]
#     
#     df_not_taken_paths_constr_infl = pd.merge(df_not_taken_paths_short,df_points_non_zero_infl, on=['Source', 'Sink'], how = 'left',suffixes=('', '_right'))
#     # get all constr names
#     list_constraints_pfl = [x for x in df_not_taken_paths_constr_infl.columns]
#     # get data in form ['Source', 'Sink', 'iLoop', 'Reason', 'Constraint', 'Influence']
#     df_not_taken_paths_constr_infl_fin = pd.melt(df_not_taken_paths_constr_infl.fillna(''), id_vars=short_cols, value_vars=list_constraints_pfl, var_name='Constraint', value_name='Influence')
#     df_not_taken_paths_constr_infl_fin = df_not_taken_paths_constr_infl_fin[df_not_taken_paths_constr_infl_fin.Influence.abs()>=0.01]
#     
#     df_not_taken_paths_constr_infl_fin.rename(columns={'iLoop': 'Step'}, inplace=True)
# =============================================================================
    
    # write to files for each source
    output_folder_by_source = os_path.join(output_path,'not_taken_paths','by_source')
    if not os_path.exists(output_folder_by_source):
        os_makedirs(output_folder_by_source)
    for name, group in df_not_taken_paths[short_cols].groupby('Source'):
        output_file = os_path.join(output_folder_by_source, name+'_not_taken_paths.csv')
        group.to_csv(output_file, index=False,columns=columns_full_file, float_format='%.3f')
        
    # write to files for each constraint
# =============================================================================
#     output_folder_by_constraint = os_path.join(output_path,'not_taken_paths','by_constraint')
#     if not os_path.exists(output_folder_by_constraint):
#         os_makedirs(output_folder_by_constraint)
#     for name, group in df_not_taken_paths_constr_infl_fin.groupby('Constraint'):
#         if ('/' in name) | ('\\' in name):
#             name = name.replace('/','_').replace('\\', '_')
#         output_file = os_path.join(output_folder_by_constraint, name+'_not_taken_paths.csv')
#         group.to_csv(output_file, index=False,columns=columns_full_file, float_format='%.3f')
# =============================================================================

def create_neg_F_split(output_path, list_constraints_for_calc_F, df_points_negF, df_constraints_table):
    print("Start neg F files",  pd.to_datetime('now'))
    fold_F_neg = os_path.join(output_path,"F_neg")
    if not os_path.exists(fold_F_neg):
        os_makedirs(fold_F_neg)
                
    df_points_negF = df_points_negF[['Source', 'Sink', 'Price'] + list(list_constraints_for_calc_F)]
    all_sources = set(df_points_negF.Source)
    constr_info = df_constraints_table[df_constraints_table.conID.isin(list_constraints_for_calc_F)][['conID','MonthSumMin','MonthSumMax','Possibility']]
    constr_info.rename(columns = {'conID':'Constraint'}, inplace = True)
    constr_info['FVmin(=sum_min*pos)'] = constr_info.MonthSumMin * constr_info.Possibility
    constr_info['FVmax(=sum_max*pos)'] = constr_info.MonthSumMax * constr_info.Possibility
    
    #split to groups by sorce and write to files
    base_columns = ['Constraint', 'FVmin(=sum_min*pos)','FVmax(=sum_max*pos)','MonthSumMin','MonthSumMax','Possibility']
    for source in list(all_sources):
        file_name = os_path.join(fold_F_neg, source+".csv")
        df_points_Source = df_points_negF[df_points_negF.Source == source]
        df_points_Source = pd.melt(df_points_Source, id_vars=['Source', 'Sink', 'Price'], value_vars=list_constraints_for_calc_F, var_name='Constraint', value_name='Influence')
        df_points_Source['Source-Sink(Price)'] = df_points_Source.Source + '-' + df_points_Source.Sink + '(' + df_points_Source.Price.astype(np.str) + ')'
        df_points_Source = df_points_Source.pivot(index='Constraint', columns='Source-Sink(Price)')['Influence'].reset_index()
        df_points_Source = df_points_Source.merge(constr_info)
        source_cols = list(df_points_Source.columns[df_points_Source.columns.str.contains(source)])
        df_points_Source[base_columns+source_cols].to_csv(file_name, index=0, float_format='%.3f')
    print("End neg F files",  pd.to_datetime('now'))

# final constraints and portfolio files
def create_final_result_files(iso, portfolio_file, constr_file, constraints_table_file, bid_class, class_mode, IS_OPTION=False):
    
# --------------- portfolio part --------------
    
    df_portfolio_data = pd.read_csv(portfolio_file)

    df_final_bids = pd.DataFrame(columns='Source,Sink,Period,Class,Instrument,Transaction,MWs,Price,Fair Value,Portfolio,Comment,FTRID'.split(','))
    df_final_bids.Source = df_portfolio_data.Source
    df_final_bids.Sink = df_portfolio_data.Sink
    df_final_bids.Period = df_portfolio_data.Step
    df_final_bids.Class = bid_class
    df_final_bids['Fair Value'] = df_portfolio_data.FV
    df_final_bids.MWs   = df_portfolio_data.MWs
    df_final_bids.Price = df_portfolio_data.Price.astype(np.int16)
    df_final_bids.Transaction = 'Buy'
    df_final_bids.Instrument = 'Obligation'
    if (iso == 'MISO'):
        # for MISO wrize path zone to Portfolio column
        # get trad points to zone file
        dict_dashboard = get_dashboard()
        filePointsToZone = dict_dashboard['TradablePointsToZoneFile']
        df_points_to_zone = pd.read_csv(filePointsToZone)
        dict_points_to_zone = pd.Series(df_points_to_zone.ExpertZone.values,index=df_points_to_zone.Name).to_dict()
        # fill portfolio column
        df_final_bids.Portfolio = df_final_bids.Source.map(dict_points_to_zone)
        df_final_bids.loc[df_final_bids.Portfolio.isnull(), 'Portfolio'] = df_final_bids.Sink.map(dict_points_to_zone) 
        df_final_bids.loc[df_final_bids.Portfolio.isnull(), 'Portfolio'] = 'Unknown'        
    else:        
        # for other iso write capital colateral (colateral*mw)
        df_final_bids.Portfolio = df_portfolio_data.Capital_Collateral
    
    # for options make only short bids file (since full neg and pos file is too big)
    if IS_OPTION:
        df_final_bids.Instrument = 'Option'
        df_final_bids.to_csv(portfolio_file.replace('.csv','_{}.csv'.format('bids')),index=0)
        return
    
    df_final_bids['Stage'] = df_portfolio_data.Stage.apply(lambda x: ';'.join(set(str(x).split(';'))).strip(';'))
    # -- full neg pos file part -----
    
    # get list of constraints which present in portfolio file
    list_constraints = [x.replace('mws_','') for x in df_portfolio_data.columns if x.find('mws_')>-1]
    df_portfolio_data.Take_constraints = df_portfolio_data.Take_constraints.fillna('') + df_portfolio_data.Take_constraints_byOneNode.fillna('')
    df_neg_pos = pd.melt(df_portfolio_data, id_vars=['Source', 'Sink', 'MWs', 'Price', 'Capital_Collateral', 'Step', 'flagStop','FV', 'Comments_neg', 'Take_constraints','Stage'], value_vars=list_constraints)
    
    df_neg_pos.columns = ['Source', 'Sink', 'MWs', 'Price', 'Capital_Collateral', 'Step', 'flagStop','FV', 'Comments_neg', 'Main', 'Stage', 'ConID', 'Influence']
    df_neg_pos['MWs_Influence'] = df_neg_pos.MWs*df_neg_pos.Influence
    df_neg_pos.Stage = df_neg_pos.Stage.apply(lambda x: ';'.join(set(str(x).split(';'))).strip(';'))
    
    # read NN table    
    df_constraints_table = get_constr_table_data(iso, class_mode, constraints_table_file)
   
    df_constraints_table['For_Comment'] = df_constraints_table.Main + ' ' + df_constraints_table.Strong
    df_constraints_table.loc[df_constraints_table.OldStatus.notnull(),'For_Comment'] = df_constraints_table.For_Comment+' '+df_constraints_table.OldStatus
    df_constraints_table.loc[df_constraints_table.For_Comment.isnull(),'For_Comment'] = df_constraints_table.OldStatus
    
    dict_main_constraints = pd.Series(df_constraints_table.For_Comment.values,index=df_constraints_table.conID).to_dict()    
    dict_FV_constraints = pd.Series(df_constraints_table.MonthSumMin.values,index=df_constraints_table.conID).to_dict()
    dict_Possib_constraints = pd.Series(df_constraints_table.Possibility.values,index=df_constraints_table.conID).to_dict()
    
    df_neg_pos['Comment'] = df_neg_pos.ConID.map(dict_main_constraints)
    df_neg_pos['FV_constr'] = df_neg_pos.ConID.map(dict_FV_constraints)*df_neg_pos.ConID.map(dict_Possib_constraints)
    df_neg_pos['FV_Influence'] = df_neg_pos.FV_constr*df_neg_pos.Influence
    df_neg_pos[['ConstraintDevice', 'ContingencyDevices', 'Direction']] = df_neg_pos.ConID.str.split(r'+',expand=True)
    df_neg_pos.to_csv(portfolio_file.replace('.csv','_{}.csv'.format('source_sink_constr')), columns = ['Source', 'Sink', 'MWs', 'Price', 'Capital_Collateral', 'ConID', 'ConstraintDevice', 'ContingencyDevices', 'Direction', 'Influence', 'MWs_Influence','FV_constr','FV_Influence', 'Main', 'Comment', 'Step', 'flagStop', 'Comments_neg', 'Stage'], index=0)
    
    # -- add take/against comment part to _bids file----
    df_neg_pos_sorted = df_neg_pos.sort_values('Influence', ascending=False)
    df_neg_pos_sorted.ConstraintDevice = df_neg_pos_sorted.ConstraintDevice + '('+(df_neg_pos_sorted.Influence*100).round().astype(str)+'%)'
    df_neg_pos_sorted['NegConstraintDevice'] = df_neg_pos_sorted.ConstraintDevice
    df_neg_pos_sorted.Main = df_neg_pos_sorted.Main.fillna('')
    df_neg_pos_sorted.ConstraintDevice = df_neg_pos_sorted.ConstraintDevice.fillna('')
    df_neg_pos_sorted.NegConstraintDevice = df_neg_pos_sorted.NegConstraintDevice.fillna('')    
    
    df_neg_pos_sorted_FV = df_neg_pos.sort_values('FV_Influence', ascending=False)
    df_neg_pos_sorted_FV.ConstraintDevice = df_neg_pos_sorted.ConstraintDevice + '('+(df_neg_pos_sorted.Influence*100).round().astype(str)+'%)'
    df_neg_pos_sorted_FV['NegConstraintDevice'] = df_neg_pos_sorted.ConstraintDevice
    df_neg_pos_sorted_FV.Main = df_neg_pos_sorted.Main.fillna('')
    df_neg_pos_sorted_FV.ConstraintDevice = df_neg_pos_sorted.ConstraintDevice.fillna('')
    df_neg_pos_sorted_FV.NegConstraintDevice = df_neg_pos_sorted.NegConstraintDevice.fillna('')    
    ###tut 4erez dict (sumConstr) DF with new sort 
    df_neg_pos_sorted_gr1 = df_neg_pos_sorted.groupby(['Source', 'Sink'], sort=False).agg({'Main':get_main_constr,'Capital_Collateral':sum}).reset_index()
    df_neg_pos_sorted_gr2 = df_neg_pos_sorted_FV[df_neg_pos_sorted.Influence > 0].groupby(['Source', 'Sink'], sort=False).agg({'ConstraintDevice':get_constr_max_3items}).reset_index()
    df_neg_pos_sorted_gr3 = df_neg_pos_sorted_FV[df_neg_pos_sorted.Influence < 0].groupby(['Source', 'Sink'], sort=False).agg({'NegConstraintDevice':get_constr_min_3items}).reset_index()
    
    df_neg_pos_sorted_gr =  pd.merge(df_neg_pos_sorted_gr1,pd.merge(df_neg_pos_sorted_gr2, df_neg_pos_sorted_gr3, how='outer'), how='outer').fillna('')
    
    df_neg_pos_sorted_gr.loc[df_neg_pos_sorted_gr.ConstraintDevice == '', 'ConstraintDevice'] = 'None'
    df_neg_pos_sorted_gr['MainTake'] = df_neg_pos_sorted_gr.ConstraintDevice
    df_neg_pos_sorted_gr.loc[df_neg_pos_sorted_gr.Main != '', 'MainTake'] = df_neg_pos_sorted_gr.Main
    
    df_neg_pos_sorted_gr.loc[df_neg_pos_sorted_gr.NegConstraintDevice == '', 'NegConstraintDevice'] = 'None'
    df_neg_pos_sorted_gr['Comment'] = 'Take: [' + df_neg_pos_sorted_gr.MainTake + '] Against: [' + df_neg_pos_sorted_gr.NegConstraintDevice + ']'
    # SourceSink to take/against comment dict
    dict_comment = pd.Series(df_neg_pos_sorted_gr.Comment.values,index=(df_neg_pos_sorted_gr.Source+df_neg_pos_sorted_gr.Sink)).to_dict()
    
    # apply dict
    df_final_bids['Comment'] = (df_final_bids.Source + df_final_bids.Sink).map(dict_comment) + 'FV='+ df_final_bids['Fair Value'].round().astype(str) + '; Step='+ df_final_bids['Period'].round().astype(str)
    df_final_bids.Comment = df_final_bids.Comment+';'+df_final_bids.Stage 
    df_final_bids.Period = ''
    df_final_bids.to_csv(portfolio_file.replace('.csv','_{}.csv'.format('bids')),index=0)

#----------- constraints part -------------------------------
    df_final_constraints = pd.DataFrame()
    df_constr_data = pd.read_csv(constr_file)
    
    df_final_constraints['conID']=df_constr_data.conID
    df_final_constraints['Dipoles']=df_constr_data.Dipoles
    df_final_constraints['Avg Coef']=df_constr_data['Avg Coef']
    df_final_constraints['Avg Neg. Coef']=df_constr_data['Avg Neg. Coef']
    df_final_constraints['NegativePaths']=df_constr_data.NegativePaths    
    
    df_final_constraints = pd.merge(df_final_constraints, df_constraints_table, how='left', on='conID')
    
    columns = 'ConstraintDevice,ContingencyDevice,ContingID,Class,Zone,Dipoles,DaySum,MonthSumMin,MonthSumMax,Duration,Possibility,N-,N+,Avg Coef,Avg Neg. Coef,ConstrType,DisappearStatus,OldStatus,Main,NegativePaths,Constancy%,Frequency%,RT%,DA%,St1,St2,kV1,kV2,Direction,constrID,Comments,Strong,DD,Drivers'.split(',')
    df_final_constraints.to_csv(constr_file.replace('.csv','_{}.csv'.format('final')),columns=columns, index=0)
    
    
def get_main_constr(x):
    # get devices form constr id
    main_constr_dev = [main_id.strip().split('+')[0] for main_id in x[:1].values[0].split(r');') if main_id.strip() != '']
    main_constr_percents = [main_id.strip().split('(')[-1] for main_id in  x[:1].values[0].split(r');') if main_id.strip() != '']
    return '; '.join(set([ pair[0]+'('+pair[1]+')' for pair in zip(main_constr_dev, main_constr_percents)]))

def get_constr_max_3items(x):
    return '; '.join(list(x.drop_duplicates())[:1])

def get_constr_min_3items(x):
    return '; '.join(list(x.drop_duplicates())[-1:])