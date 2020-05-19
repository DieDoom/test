import sys
from os import path as os_path
#from os import path as os_path, makedirs as os_makedirs, remove as os_remove, environ as os_environ
from shutil import copy as shutil_copy
from copy import deepcopy as copy_deepcopy

folderScript   = os_path.dirname(sys.argv[0])
folderMain     = os_path.dirname(folderScript)
folderSnippets = os_path.join(folderMain,'snippets')
if not os_path.exists(folderSnippets):
    folderSnippets = r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\snippets'
sys.path.insert(0,folderSnippets)
sys.path.insert(0, r"z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\pathsOptimizer")
from influence_lib_loader import get_influence_library_to_dict_with_additional_lib
from po_system_functions import get_common_root, df_empty, LogStageValue, create_clear_folder
from po_data_load_functions import get_constr_table_data, get_source_sink_file_data, get_shadow_price_data, get_paths_to_use, get_start_portfolio
from po_main_calc_functions import calc_Price, calc_Collateral, calc_FV, calc_F_path0, fix_F_for_Nplus_excess, get_corr_clusters, choose_cur_main_constr, finalize_loop, calc_weight_path
from po_output_functions import create_final_not_taken_paths_files, create_neg_F_split, write_portfolio_file, write_constraints_file, create_final_result_files
sys.path.insert(0, r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\risk_management")
from risk_management import add_constr_influence_columns, calc_RM_coef_portfolio, check_Nminus_violation
sys.path.insert(0,r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\standard_functions')
from market_standard_functions import get_hubs

#from options import COptions
from timerClock import TimerClock
from fileIO import ReadOnly
timer = TimerClock()

import numpy as np
import pandas as pd
from itertools import combinations


dict_class_mode_to_name = {1:'OnPeak', 0:'OffPeak', 2:'Peak WE'}

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def run_path_optimizer(CLUSTERS_iso, CLUSTER_PATH_ROOT, class_mode, CLUSTERS_constr_table, CLUSTERS_sourcesink, IS_REMOVE_HUBS, CLUSTERS_RCTool, coef_sigma_down, ignor_3minFTR_for_down_price, min_corr_value, CLUSTERS_influence_lib, CLUSTERS_influence_lib_additional, n_plus_step, CLUSTERS_shadow_price, CLUSTERS_price_distribution_file,min_price_delta, min_price_fraction_delta, CLUSTER_collateral, IS_UNITARY_COLLATERAL, min_collateral_value, min_add_price, price_percent, min_value_price, min_pos_influence_to_clear, min_neg_influence_to_clear, IS_CALC_PRICE_FROM_MEAN_SIGMA, IS_OPTION, FILE_CONSTRAINTS_FOR_FV, IS_REMOVE_NODES, FILE_NODES_TO_REMOVE, IS_ANNUAL_PRICES,  ANNUAL_PRICE_MODE, IS_USE_PATHS_FROM_FILE, SOURCE_SINK_TO_USE_FILE, IS_DIVIDE_QUARTAL_PRICES, IS_OFF_STAGE_R, CLUSTER_F_weight_coef, IS_WRITE_BASE_TO_DISK, hack_take_only_main_constr_paths, max_count_nodes, min_influence_for_node, min_influence_for_path, path_to_fit_EESL, use_EESL, hack_FV_BOT_limit, hack_FV_BOT_value, hack_price_TOP_limit, hack_price_TOP_value, hack_price_BOT_limit, hack_price_BOT_value, hack_collat_TOP_limit, hack_collat_TOP_value, hack_collat_BOT_limit, hack_collat_BOT_value, DO_F_NEG_SPLIT_OUTPUT, MWs_step, hack_take_portfolio_from_file, start_portfolio_file, iLoopMax, IS_CHOOSE_CONSTR_BY_FV, IS_WRITE_NEG_CONSTR_COMMENT, mws_max_path, calc_F_main, mws_max_node, minChange, coef_recent_neg, IS_CHECK_MinDART_RECENT, DO_FULL_OUTPUT_FILES, IS_DOWN_PRICE=False, USE_MIN_RT_DA=False, min_RT=0, min_DA=0):
    
    IS_CALC_PRICE_FROM_REF_PRICE = False
    
    timer.start('check')
    
    create_clear_folder(CLUSTER_PATH_ROOT)
    
    LogStageValue('Input parameters: ', CLUSTER_PATH_ROOT, "id")
    for name, value in locals().items():
        LogStageValue('{} = {}'.format(name, value), CLUSTER_PATH_ROOT, "id")
    
    # load NN table
    print('loading constr table')
    df_constraints_table = get_constr_table_data(CLUSTERS_iso, class_mode, CLUSTERS_constr_table)
    print('finish loading constr table')

    # looad source sink file
    df_points_sourcesink_source = get_source_sink_file_data(CLUSTERS_iso, class_mode, CLUSTERS_sourcesink, IS_REMOVE_HUBS)   
    
# =============================================================================
    iLoop = 0.07
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points_sourcesink_source.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    # get clusters
    timer.start('corr')
    (list_points_unique, dict_rename) = get_corr_clusters(CLUSTERS_iso, dict_class_mode_to_name[class_mode], CLUSTERS_RCTool, min_corr_value, df_points_sourcesink_source,  get_common_root(CLUSTER_PATH_ROOT))
    timer.end('corr','done getting clusters  ')

    # load inf lib
    print('loading influence lib')
    if 'PriceNode' in df_points_sourcesink_source.columns:
        df_hubs = get_hubs(df_points_sourcesink_source)
        hubs_list = list(set(df_hubs.Name))
    else:    
        hubs_list = []
        
    # for ISONE do not remove anything
    if CLUSTERS_iso == 'ISONE':
        hubs_list = []
    
    is_clear_lib_inf_cash = False
    is_return_second_dict = False
    
    if (not os_path.isfile(CLUSTERS_influence_lib_additional)) & (not os_path.isdir(CLUSTERS_influence_lib_additional)):
        dict_points_influence = get_influence_library_to_dict_with_additional_lib(CLUSTERS_iso, CLUSTERS_influence_lib, CLUSTERS_influence_lib_additional, hubs_list, list(df_constraints_table.conID), get_common_root(CLUSTER_PATH_ROOT), is_clear_lib_inf_cash, is_return_second_dict)
        is_return_second_dict = True
        dict_points_influence_ID_const = get_influence_library_to_dict_with_additional_lib(CLUSTERS_iso, CLUSTERS_influence_lib, CLUSTERS_influence_lib_additional, hubs_list, list(df_constraints_table.conID), get_common_root(CLUSTER_PATH_ROOT), is_clear_lib_inf_cash, is_return_second_dict)
        
    else: 
        dict_points_influence = get_influence_library_to_dict_with_additional_lib(CLUSTERS_iso, CLUSTERS_influence_lib, CLUSTERS_influence_lib_additional, hubs_list, list(df_constraints_table.conID), get_common_root(CLUSTER_PATH_ROOT), is_clear_lib_inf_cash, is_return_second_dict)
        dict_points_influence_ID_const = dict_points_influence
    print('finish loading influence lib')
    
    # create keys of excisting inf data in form constr+conting+dir
    # dict_points_influence.keys() - 'node+constr+conting+dir'
    list_inf_constr = list(set(dict_points_influence.keys()))
    list_inf_constr = [x.split('+') for x in list_inf_constr]
    list_inf_constr = [x[1]+'+'+x[2]+'+'+x[3] for x in list_inf_constr]

    list_available_constraints_dipoles = list(set(list_inf_constr))
    df_constraints_table_not_found = df_constraints_table[~df_constraints_table.conID.isin(list_inf_constr)]
    df_constraints_table_not_found.to_csv(get_common_root(CLUSTER_PATH_ROOT,r'Constraints_without_dipoles_{}.csv'.format(df_constraints_table_not_found.shape[0])),index=0)
    
    list_constraints_from_NN_table = df_constraints_table.conID.unique().tolist()
    list_All_constraints_from_NN_table = list(set(list_constraints_from_NN_table))
    dict_coeff_constraints_from_NN_table = dict()                         
    # set dict key = Constraint+Contingency+Direction and value = list [[N-,N+],[sumMin, sumMax],Possibility,Main,OldStatus,DisappearStatus,Seasonality,ConstraintDevice,ContingencyDevice,FV]
    for line in df_constraints_table[['ConstraintDevice','ContingencyDevice','N-','N+','MonthSumMin', 'MonthSumMax', 'Possibility', 'Direction', 'Main', 'OldStatus', 'DisappearStatus', 'Seasonality']].values:
        constr_contin_from_NN_table = line[0]+'+'+line[1]+'+'+str(line[7])    
        dict_coeff_constraints_from_NN_table[constr_contin_from_NN_table] = [[float(line[2]),float(line[3])],[float(line[4]),float(line[5])],float(line[6])/100,str(line[8]),str(line[9]),str(line[10]),str(line[11]),str(line[0]),str(line[1]),float((line[4])*(line[6])/100)]

    df_constraints_table = df_constraints_table[df_constraints_table.conID.isin(list_inf_constr)]

    list_constraints = df_constraints_table.conID.unique().tolist()
    list_main_constraints = df_constraints_table[df_constraints_table.Main.fillna('').str.contains('MainConstr')].conID.tolist()
    # remove 5min and "clear" constraints from main
    list_main_constraints = [constr for constr in list_main_constraints if ('5min' not in constr  and '_clear' not in constr)]

    list_All_constraints = list(set(list_constraints))
    dict_main_constraints = pd.Series(df_constraints_table.For_Comment.values,index=df_constraints_table.conID).to_dict()
    
    dict_coeff_constraints = dict()                         
    # set dict key = Constraint+Contingency+Direction and value = list [[N-,N+],[sumMin, sumMax],Possibility,Main,OldStatus,DisappearStatus,Seasonality,ConstraintDevice,ContingencyDevice,FV]
    for line in df_constraints_table[['ConstraintDevice','ContingencyDevice','N-','N+','MonthSumMin', 'MonthSumMax', 'Possibility', 'Direction', 'Main', 'OldStatus', 'DisappearStatus', 'Seasonality']].values:
        constr_contin = line[0]+'+'+line[1]+'+'+str(line[7])    
        dict_coeff_constraints[constr_contin] = [[float(line[2]),float(line[3])],[float(line[4]),float(line[5])],float(line[6])/100,str(line[8]),str(line[9]),str(line[10]),str(line[11]),str(line[0]),str(line[1]),float((line[4])*(line[6])/100)]

    max_N_plus = max([coef[0][1] for coef in dict_coeff_constraints.values()])    

    dict_coeff_constraints_orig_table = copy_deepcopy(dict_coeff_constraints)

    # hack set all N+ limits for main constraints to n_plus_step
    for constr in list_main_constraints:
        dict_coeff_constraints[constr][0][1] = n_plus_step

    # set final list with Constraint,Contingency,Direction
    list_All_constraints = list(dict_coeff_constraints.keys())

    # set columns names in future dataframe to fast get
    list_All_mws_constraints = ['mws_'   + constr for constr in list_All_constraints_from_NN_table]

    dict_constr_in_dipoles = dict()
    for constr in list_All_constraints_from_NN_table:
        dict_constr_in_dipoles[constr] = '+' if constr in list_available_constraints_dipoles else 'NO'
        
    iLoop = 0.075
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    CLUSTER_F_correction_coef = 1.0
    (df_points, dict_rename, list_constr_for_FV, df_points_negF, min_recent_delta, coef_recent_neg_stageR) = get_base_df_points_array(list_All_constraints, list_main_constraints, dict_points_influence_ID_const, dict_coeff_constraints, CLUSTER_PATH_ROOT, coef_sigma_down, ignor_3minFTR_for_down_price, IS_OFF_STAGE_R, CLUSTERS_iso, class_mode, calc_F_main, CLUSTERS_sourcesink, list_points_unique, dict_rename, list_constraints, CLUSTERS_shadow_price, CLUSTERS_price_distribution_file,min_price_delta, min_price_fraction_delta, CLUSTER_collateral, IS_UNITARY_COLLATERAL, min_collateral_value, min_add_price,price_percent,min_value_price, min_pos_influence_to_clear, min_neg_influence_to_clear, path_to_fit_EESL, use_EESL, IS_CALC_PRICE_FROM_MEAN_SIGMA, IS_OPTION, FILE_CONSTRAINTS_FOR_FV,IS_REMOVE_NODES, FILE_NODES_TO_REMOVE, IS_REMOVE_HUBS, IS_ANNUAL_PRICES,  ANNUAL_PRICE_MODE, IS_USE_PATHS_FROM_FILE, SOURCE_SINK_TO_USE_FILE, IS_DIVIDE_QUARTAL_PRICES, IS_CALC_PRICE_FROM_REF_PRICE, CLUSTER_F_correction_coef, CLUSTER_F_weight_coef, IS_DOWN_PRICE, USE_MIN_RT_DA,  min_RT, min_DA, is_return_second_dict)
    

    
    save2_columns = ['Source','Sink','Source1','Sink1','MWs','Capital','flagStop','Comments','Comments_neg','Comments_coef','Take_constraints','Take_constraints_byOneNode','Fp_i','FV','Fp_i0','Price','Fp_i03','Price3', 'Fp_iR','Collateral','Collateral_t','iLoop','Price_0','Price3_0', 'Weight']
        
    df_points_negF.to_csv(get_common_root(CLUSTER_PATH_ROOT,r'save2\{}_{}_b_Fp_i_less_zero.csv'.format(0,"id")),index=0,columns=save2_columns, float_format='%g')

# =============================================================================
    iLoop = 0.08
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points.to_csv(root_array_df)
# =============================================================================

    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')

    # export paths with F < 0 with constr info (constr, possib, sum, FV(=sum*possib), inf_to_path) 
    #  to into files splitted by source name
    if DO_F_NEG_SPLIT_OUTPUT: 
        list_constraints_for_calc_F = dict_coeff_constraints.keys()
        if calc_F_main == 1: # only for main constr
            list_constraints_for_calc_F = list_main_constraints
        create_neg_F_split(CLUSTER_PATH_ROOT, list_constraints_for_calc_F, df_points_negF, df_constraints_table)
# =============================================================================
#     if IS_WRITE_BASE_TO_DISK:
#         file_base_file = os_path.splitext(os_path.basename(CLUSTERS_base))[0].split('+')[0]     
#         full_rows = df_points.shape[0]
#         new_base_file = os_path.join(CLUSTER_PATH_ROOT,'{}+{}.csv'.format(file_base_file,full_rows))
#         
#         print('start write to disk base file')
#         df_points.to_csv(new_base_file, index=False, float_format='%g')
#         writeStage('CSV_base_file', CLUSTER_PATH_ROOT, CLUSTER_ID)
#         
#         raise ValueError('End of creating points DataFrame')
# =============================================================================
        
    # hack take paths which have pos influence to main constr
    if hack_take_only_main_constr_paths:
        cond_main_constr_paths = df_points[list_main_constraints[0]]>0
        for constr in list_main_constraints[1:]:
            cond_main_constr_paths = cond_main_constr_paths | df_points[constr]>0
            df_points.loc[~cond_main_constr_paths, 'Comments'] = 'dont take any main constraint'
            df_points[~cond_main_constr_paths].to_csv(get_common_root(CLUSTER_PATH_ROOT,r'save2\{}_{}_a.csv'.format(0,"id")),index=0,columns=['Source','Sink','Source1','Sink1','MWs','Capital','flagStop','Comments','Comments_neg','Comments_coef','iLoop'], float_format='%g')
            df_points = df_points[cond_main_constr_paths  | (df_points.MWs>0)]
    
    # fill take constr columns        
    df_points['Take_constraints'] = '' # columns with constr names, delim ;
    df_points['Take_constraints_byOneNode'] = ''
    df_points['is_non_zero_infl'] = 0

    df_points = fill_take_constr_columns(df_points, dict_points_influence, list_main_constraints, max_count_nodes, min_influence_for_node, min_influence_for_path, dict_rename, get_common_root(CLUSTER_PATH_ROOT, 'list_ConstrID_points_clear.csv'))
    
    df_points_non_zero_infl = df_points[df_points['is_non_zero_infl']==1]

    # hack remove paths with FV > hack_FV_BOT_value
    if hack_FV_BOT_limit:
        cond_fv_bot_limit = df_points.FV < hack_FV_BOT_value
        df_points.loc[cond_fv_bot_limit, 'Comments'] = 'FV < {}'.format(hack_FV_BOT_value)
        df_points.loc[cond_fv_bot_limit, 'flagStop'] = 15
        df_points[cond_fv_bot_limit].to_csv(get_common_root(CLUSTER_PATH_ROOT, r'save2\{}_{}_a.csv'.format("FV_lim","id")),index=0,columns=save2_columns, float_format='%g')
        df_points = df_points[(~cond_fv_bot_limit) | (df_points.MWs>0)]

    # hack remove paths with price > hack_price_TOP_value
    if hack_price_TOP_limit:
        cond_price_top_limit = df_points.Price > hack_price_TOP_value
        df_points.loc[cond_price_top_limit, 'Comments'] = 'Price > {}'.format(hack_price_TOP_value)
        df_points.loc[cond_price_top_limit, 'flagStop'] = 16
        df_points[cond_price_top_limit].to_csv(get_common_root(CLUSTER_PATH_ROOT, r'save2\{}_{}_a.csv'.format("Price_lim_top","id")),index=0,columns=save2_columns, float_format='%g')
        df_points = df_points[(~cond_price_top_limit) | (df_points.MWs>0)]
    
    # hack remove paths with price < hack_price_BOT_value
    if hack_price_BOT_limit:
        cond_price_bot_limit = df_points.Price < hack_price_BOT_value
        df_points.loc[cond_price_bot_limit, 'Comments'] = 'Price < {}'.format(hack_price_BOT_value)
        df_points.loc[cond_price_bot_limit, 'flagStop'] = 17
        df_points[cond_price_bot_limit].to_csv(get_common_root(CLUSTER_PATH_ROOT, r'save2\{}_{}_a.csv'.format("Price_lim_bot","id")),index=0,columns=save2_columns, float_format='%g')
        df_points = df_points[(~cond_price_bot_limit) | (df_points.MWs>0)]
        
    # hack remove paths with collateral > hack_collat_TOP_value
    if hack_collat_TOP_limit:
        cond_collat_top_limit = df_points.Collateral_t > hack_collat_TOP_value
        df_points.loc[cond_collat_top_limit, 'Comments'] = 'Collateral > {}'.format(hack_collat_TOP_value)
        df_points.loc[cond_collat_top_limit, 'flagStop'] = 18
        df_points[cond_collat_top_limit].to_csv(get_common_root(CLUSTER_PATH_ROOT, r'save2\{}_{}_a.csv'.format("Collat_lim_top","id")),index=0,columns=save2_columns, float_format='%g')
        df_points = df_points[(~cond_collat_top_limit) | (df_points.MWs>0)]
    
    # hack remove paths with collateral < hack_collat_BOT_value
    if hack_collat_BOT_limit:
        cond_collat_bot_limit = df_points.Collateral_t < hack_collat_BOT_value
        df_points.loc[cond_collat_bot_limit, 'Comments'] = 'Collateral < {}'.format(hack_collat_BOT_value)
        df_points.loc[cond_collat_bot_limit, 'flagStop'] = 19
        df_points[cond_collat_bot_limit].to_csv(get_common_root(CLUSTER_PATH_ROOT, r'save2\{}_{}_a.csv'.format("Collat_lim_bot","id")),index=0,columns=save2_columns, float_format='%g')
        df_points = df_points[(~cond_collat_bot_limit) | (df_points.MWs>0)]
    
    # set types for columns
    df_points.FV = df_points.FV.astype(np.float64)
    df_points.Collateral =  df_points.Collateral.astype(np.float16)


    df_points['Comments'] = ''
    df_points['Comments_neg'] = ''
    df_points['Comments_coef'] = ''

    df_points['iLoop'] = 1

    df_points['Stage'] = ''

    # no loops for options - just create final output files with paths with positive F
# =============================================================================
#     if IS_OPTION:
#         df_points.MWs = 1
#         for constr in list_All_constraints:
#             if constr in df_points.columns:
#                 df_points['mws_'+constr] = df_points[constr]*df_points.MWs
#         (dict_coef_portfolio, dict_coef_portfolio_neg) = calc_RM_coef_portfolio(dict_coeff_constraints.keys(), df_points)
#         df_points['Capital_Collateral'] = (df_points.MWs*df_points.Collateral_t).astype(np.float32)
#         write_portfolio_file(get_common_root(CLUSTER_PATH_ROOT,r'common'), 0, df_points, df_points.columns.tolist())
#         write_constraints_file(get_common_root(CLUSTER_PATH_ROOT,r'common'), 0, df_points, list_All_constraints, list_All_mws_constraints, dict_constr_in_dipoles, dict_coef_portfolio, dict_coef_portfolio_neg, dict_main_constraints, dict_coeff_constraints)
#         bid_class = dict_class_mode_to_name[class_mode]
#             
#         create_final_result_files(CLUSTERS_iso, os_path.join(get_common_root(CLUSTER_PATH_ROOT,r'common'),'portfolio_{}.csv'.format(0)), os_path.join(get_common_root(CLUSTER_PATH_ROOT,r'common'),'constraints_{}.csv'.format(0)) , CLUSTERS_constr_table, bid_class, class_mode, IS_OPTION)
#         
#         shutil_copy(CLUSTERS_constr_table, get_common_root(CLUSTER_PATH_ROOT,''))
#         if IS_CALC_PRICE_FROM_MEAN_SIGMA:
#             shutil_copy(CLUSTERS_price_distribution_file, get_common_root(CLUSTER_PATH_ROOT,''))
#         # write output file for untaken paths
#         timer.start('1')
#         create_final_not_taken_paths_files(CLUSTERS_iso, get_common_root(CLUSTER_PATH_ROOT,''), get_common_root(CLUSTER_PATH_ROOT,r'save2'),    df_points_non_zero_infl,dict_coeff_constraints.keys(), MWs_step, dict_coeff_constraints_orig_table)
#         timer.end('1', 'end of write file')         
#     
#         raise ValueError("that's all folks")    
# =============================================================================
# =============================================================================
    iLoop = 0.09
#     
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    # get bad price paths to move them to 3d stage
    if IS_CALC_PRICE_FROM_MEAN_SIGMA:
        df_points_bad_price = df_points[(df_points.max_F_ind == -1) & (df_points.MWs == 0)]
        df_points = df_points[~(df_points.max_F_ind == -1) | (df_points.MWs > 0)]
    else:
        df_points_bad_price = pd.DataFrame()
    
# =============================================================================
    iLoop = 0.1
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    # split df_points array to df_points - which are have non empty value in Take_constraints field and df_ponits_no_take_constraints
    df_points_take_constraints_by_one_node = df_points[(df_points['Take_constraints_byOneNode']!="") & (df_points['Take_constraints']=="") & (df_points.MWs == 0) & (df_points.flagStop != 10)]
    df_points_no_take_constraints = df_points[((df_points['Take_constraints']=="") & (df_points['Take_constraints_byOneNode']=="") & (df_points.MWs == 0)) | (df_points.flagStop == 10)]
    # for start - stage one - as df_points take only paths which takes constraints with both nodes
    df_points = df_points[((df_points['Take_constraints']!="") | (df_points.MWs > 0)) & (df_points.flagStop != 10)]
    
# =============================================================================
    iLoop = 0.2
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    #----- Take start portfolio from file ---------   

    # Source Sink MWs columns should present
    if hack_take_portfolio_from_file:
        timer.start('2')
       
        (df_points_start, dict_coeff_constraints) = get_start_portfolio(start_portfolio_file, class_mode, list_constraints,  dict_coeff_constraints, dict_rename, dict_points_influence, CLUSTER_PATH_ROOT )
        df_points_start = fill_take_constr_columns(df_points_start, dict_points_influence, list_main_constraints, max_count_nodes, min_influence_for_node, min_influence_for_path, dict_rename, get_common_root(CLUSTER_PATH_ROOT, 'list_ConstrID_points_clear_st_portf.csv'))
        df_points_start['Stage'] = 'start portf'
        
        timer.end('2',"Finish loading existing portfolio")
    
        df_points = pd.concat([df_points,df_points_start], ignore_index=True, sort=False)
        
        df_points.sort_values(by=['MWs'], ascending = True, inplace = True)
        
        df_points.drop_duplicates(['Source1','Sink1'], keep='last', inplace=True)
    #----- end Take start portfolio from file --------- 
    
    # initialyze vars 
    iLoop = 1
    stage = 1
    loop_count_RM = 0
    loop_count_Fcalc = 0
    loop_count_takePath = 0
    cur_Nplus_coef = MWs_step
    
    # for output files header
    list_column_order = df_points.columns.tolist()
    
    n_main_constr = len(set(list_main_constraints))
    constraints_taken_by_limits = set()
    
    # dict_constr_FV - used for choosing constraint by FV = sumMin*possibility
    dict_constr_FV = dict()
    for constr in dict_coeff_constraints.keys():
        # [N-,N+], [sumMin, sumMax], Possibility for constr
        list_limits_constraints = dict_coeff_constraints[constr]
        dict_constr_FV[constr] = list_limits_constraints[1][0]*list_limits_constraints[2]
    
        # calculate coeff for portfolio
    (dict_coef_portfolio, dict_coef_portfolio_neg) = calc_RM_coef_portfolio(dict_coeff_constraints.keys(), df_points)
    dict_coef_portfolio_prev = copy_deepcopy(dict_coef_portfolio)
    
    root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
    df_points.to_csv(root_array_df)
    
# =============================================================================
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df1_{}.csv'.format(iLoop))
#     df_points_take_constraints_by_one_node.to_csv(root_array_df)
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df2_{}.csv'.format(iLoop))
#     df_points_no_take_constraints.to_csv(root_array_df)
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df3_{}.csv'.format(iLoop))
#     df_points_bad_price.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    while iLoop<=iLoopMax:
        res = path_optimizer_main_loop(CLUSTERS_iso, iLoop, df_points, CLUSTER_PATH_ROOT, save2_columns, stage, list_main_constraints, constraints_taken_by_limits, dict_constr_FV, IS_CHOOSE_CONSTR_BY_FV, n_main_constr, dict_coeff_constraints, dict_coeff_constraints_from_NN_table, dict_coef_portfolio, loop_count_RM, MWs_step, dict_coef_portfolio_neg, IS_WRITE_NEG_CONSTR_COMMENT, mws_max_path, loop_count_Fcalc, dict_coeff_constraints_orig_table, calc_F_main,list_constr_for_FV, mws_max_node, loop_count_takePath, list_column_order, minChange, list_All_constraints, list_All_constraints_from_NN_table, cur_Nplus_coef, n_plus_step,  max_N_plus, df_points_take_constraints_by_one_node,df_points_no_take_constraints, df_points_bad_price, dict_coef_portfolio_prev, CLUSTER_F_correction_coef, min_price_delta, DO_FULL_OUTPUT_FILES, list_All_mws_constraints, dict_constr_in_dipoles, dict_main_constraints, class_mode, CLUSTERS_constr_table, min_price_fraction_delta, coef_recent_neg, min_recent_delta, coef_recent_neg_stageR, IS_CHECK_MinDART_RECENT)
        if res == 0:
            break
        (df_points, stage, constraints_taken_by_limits, dict_coeff_constraints, dict_coef_portfolio, loop_count_RM, dict_coef_portfolio_neg, loop_count_Fcalc, calc_F_main, loop_count_takePath, cur_Nplus_coef, dict_coef_portfolio_prev, coef_recent_neg, IS_CHECK_MinDART_RECENT) = res
        iLoop+=1
    
# =============================================================================
#     root_array_df = get_common_root(CLUSTER_PATH_ROOT,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), CLUSTER_PATH_ROOT, "id")
    timer.start('check')
    
    shutil_copy(CLUSTERS_constr_table, get_common_root(CLUSTER_PATH_ROOT,''))
    if IS_CALC_PRICE_FROM_MEAN_SIGMA:
        shutil_copy(CLUSTERS_price_distribution_file, get_common_root(CLUSTER_PATH_ROOT,''))
        # write output file for untaken paths
        timer.start('1')
        create_final_not_taken_paths_files(CLUSTERS_iso, get_common_root(CLUSTER_PATH_ROOT,''), get_common_root(CLUSTER_PATH_ROOT,r'save2'),            df_points_non_zero_infl,dict_coeff_constraints.keys(), MWs_step, dict_coeff_constraints_orig_table)
        timer.end('1', 'end of write file')         
        print()
        print('Done..')




def get_base_df_points_array(list_All_constraints, list_main_constraints, dict_points_influence, dict_coeff_constraints, log_path, coef_sigma_down, ignor_3minFTR_for_down_price, IS_OFF_STAGE_R, CLUSTERS_iso, class_mode, calc_F_main, CLUSTERS_sourcesink, list_points_unique, dict_rename, list_constraints, CLUSTERS_shadow_price, CLUSTERS_price_distribution_file,min_price_delta, min_price_fraction_delta, CLUSTER_collateral, IS_UNITARY_COLLATERAL, min_collateral_value, min_add_price,price_percent,min_value_price, min_pos_influence_to_clear, min_neg_influence_to_clear, path_to_fit_EESL, use_EESL, IS_CALC_PRICE_FROM_MEAN_SIGMA, IS_OPTION=False, FILE_CONSTRAINTS_FOR_FV="",IS_REMOVE_NODES=False, FILE_NODES_TO_REMOVE="", IS_REMOVE_HUBS=False, IS_ANNUAL_PRICES=False, ANNUAL_PRICE_MODE='',IS_USE_PATHS_FROM_FILE=False, SOURCE_SINK_TO_USE_FILE='', IS_DIVIDE_QUARTAL_PRICES=False, IS_CALC_PRICE_FROM_REF_PRICE=False, CLUSTER_F_correction_coef = 1.0, CLUSTER_F_weight_coef = 1.0, IS_DOWN_PRICE=False,USE_MIN_RT_DA=False, min_RT=0, min_DA=0, is_return_second_dict = False):
    # CLUSTERS_base create    
    # get clusters nodes, clear file, make dict

    print('inside get_base_df_points_array')
    timer.start('1')
    
    df_points_sourcesink_source = get_source_sink_file_data(CLUSTERS_iso, class_mode, CLUSTERS_sourcesink, IS_REMOVE_HUBS)    
    nodes_from_source_sink_file = list(set(df_points_sourcesink_source.Name))
    
    if IS_CALC_PRICE_FROM_MEAN_SIGMA | IS_CALC_PRICE_FROM_REF_PRICE:
        df_shadow_price=pd.DataFrame()
    else:
        # shadow price from auction
        df_shadow_price = get_shadow_price_data(CLUSTERS_iso, class_mode, CLUSTERS_shadow_price, IS_ANNUAL_PRICES)
        nodes_from_shadow_price_file = list(set(df_shadow_price.SourceSink))
    
        # choose only source-sink and shadow-price data wich have nodes in common
        df_points_sourcesink_source = df_points_sourcesink_source[df_points_sourcesink_source.Name.isin(nodes_from_shadow_price_file)]
        df_shadow_price = df_shadow_price[df_shadow_price.SourceSink.isin(nodes_from_source_sink_file)]
 
        points_with_price_list = list(df_shadow_price.SourceSink)
    
        # delete not found points with price
        not_found_points = []
        for point in list_points_unique:
            if point not in points_with_price_list:
                not_found_points.append(point)
           
        for not_found_point in not_found_points:
            list_points_unique.remove(not_found_point)
    
    if IS_REMOVE_NODES:
        if not os_path.isfile(FILE_NODES_TO_REMOVE):
            raise ValueError('Can not find file for nodes to remove: {}'.format(FILE_NODES_TO_REMOVE))
        df_nodes_to_remove = pd.read_csv(FILE_NODES_TO_REMOVE)
        list_nodes_to_remove = df_nodes_to_remove.Node.to_list()
        list_points_unique = [ node for node in list_points_unique if node not in list_nodes_to_remove]        
    
    print('Unique points:',len(list_points_unique))
    # make all available pair Source - Sink from points    
    list_combination1 = list(combinations(list_points_unique, 2))
    # make reverse combinations
    list_combination2 = [(x[1],x[0]) for x in list_combination1]
    list_combination = list_combination1 + list_combination2
    
    if IS_USE_PATHS_FROM_FILE:
        list_combination = get_paths_to_use(SOURCE_SINK_TO_USE_FILE, list_combination)
    
    list_combinationA = [x[0] for x in list_combination]
    list_combinationB = [x[1] for x in list_combination]
    
    # create dataframe with headers and types
    list_cols_s     = ['Source', 'Sink', 'Price'   ,'Price3', 'MWs'  , 'MWs_prev', 'Capital' , 'Source1', 'Sink1']
    list_cols_types = [np.str  , np.str, np.float32, np.float32, np.float16, np.float16   , np.float32, np.str   , np.str]
    df_points = df_empty(list_cols_s,list_cols_types) 
    
    # set to dataframe all paths
    df_points.Source = list_combinationA
    df_points.Sink = list_combinationB
    
# =============================================================================
    iLoop = 0.077
#     root_array_df = get_common_root(log_path,r'common\Array_df_{}.csv'.format(iLoop))
#     df_points.to_csv(root_array_df)
# =============================================================================
    
    t_end1 = timer.end('check')
    LogStageValue('check {}: end in {} sec'.format(iLoop,t_end1), log_path, "id")
    timer.start('check')
    
    if CLUSTERS_iso == 'CAISO':
        df_points = remove_restricted_path(CLUSTERS_iso, df_points, df_points_sourcesink_source, log_path, path_to_fit_EESL, use_EESL, class_mode)
        
    if CLUSTERS_iso == 'SPP':
        df_points = remove_restricted_path(CLUSTERS_iso, df_points, df_points_sourcesink_source, log_path, path_to_fit_EESL, use_EESL, class_mode)
        
    if (CLUSTERS_iso == 'ERCOT') & IS_OPTION:
        #df_points = remove_restricted_path(CLUSTERS_iso, df_points, (OPTION_FTR_LMP_FILE,class_mode), log_path)
        # also add 'Instrument' column for options
        df_points['Instrument'] = 'Option'

    timer.start('inf')
    df_points['Price'] = 0.
    # add constraints and mws_constr columns

    is_throw_empty_nodes = False
    if CLUSTERS_iso == 'CAISO':
        is_throw_empty_nodes = True
    df_points = add_constr_influence_columns(list_All_constraints, df_points, dict_points_influence, log_path, is_throw_empty_nodes, is_return_second_dict)
    
    for constr in list_constraints:
        if constr in df_points.columns:
            df_points.loc[(df_points[constr]>0) & (df_points[constr].abs()<min_pos_influence_to_clear),constr] = 0
            df_points.loc[(df_points[constr]<0) & (df_points[constr].abs()<min_neg_influence_to_clear),constr] = 0
    
    timer.end('inf','done making constraints columns  ')

    print('Before calc FV ')
    timer.start('fv')
 
    full_constr_list = sorted(list(dict_coeff_constraints.keys()))
    list_constr_for_FV = full_constr_list
    
    # take constr for FV calc from exparts file
    if (calc_F_main == 4):
        if not os_path.isfile(FILE_CONSTRAINTS_FOR_FV):
            raise ValueError('Can not find file with constraint for FV calculation FILE_CONSTRAINTS_FOR_FV: {}'.format(FILE_CONSTRAINTS_FOR_FV))
        # get given constraints in form Constraint+Contingency+Direction
        constr_for_FV_data = pd.read_csv(FILE_CONSTRAINTS_FOR_FV)
         
        constr_for_FV_data.CDevices = constr_for_FV_data.CDevices.str.replace("; ",";")
        # set blank contingencies to 'BASE'
        constr_for_FV_data.loc[constr_for_FV_data.CDevices.isnull(),'CDevices'] = 'BASE'
        # fix direction
        constr_for_FV_data['Dir'] = pd.to_numeric(constr_for_FV_data['Dir'].fillna(0), errors='ignore', downcast='integer')
        constr_for_FV_data.Dir = constr_for_FV_data.Dir.astype(str)
        
        constr_for_FV_data['conID'] = constr_for_FV_data.ConstraintDeviceName.fillna('') + '+' + constr_for_FV_data.CDevices.fillna('') + '+' + constr_for_FV_data.Dir
               
        list_constr_for_FV = constr_for_FV_data['conID'].to_list()     
        
        absent_FV_constraints = [constr for constr in list_constr_for_FV if constr not in full_constr_list]
        if len(absent_FV_constraints):
            print('!!!!!!!!\n!!!!!  There are ',len(absent_FV_constraints),' constraints absent in constr table, but needed for FV calc!\n Please check it!\t!!!!!!!\n!!!!!!!')
            # write absent file 
            abs_constr_for_FV_file = os_path.join(log_path,'Absent_constr_for_FV.txt')
            with open(abs_constr_for_FV_file, 'w') as f:
                for item in absent_FV_constraints:
                    f.write("%s\n" % item)
            
            # remove absent constraints
            list_constr_for_FV = [constr for constr in list_constr_for_FV if constr not in absent_FV_constraints]
            
    df_points = calc_FV(df_points, calc_F_main, dict_coeff_constraints, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef)    
    
    #CLUSTER_F_weight_coef = 1.0
    
    
    df_points = calc_weight_path(df_points, dict_coeff_constraints, list_constr_for_FV, list_main_constraints, CLUSTER_F_weight_coef)
    
    timer.end('fv','After calc FV ')
    
    print('Before calc price ')
    timer.start('tp')
    
    df_points = calc_Price(df_points, class_mode, CLUSTERS_iso, CLUSTERS_price_distribution_file, log_path, coef_sigma_down, ignor_3minFTR_for_down_price, df_shadow_price,IS_CALC_PRICE_FROM_MEAN_SIGMA, min_price_delta, min_price_fraction_delta, min_add_price, price_percent, min_value_price, IS_CALC_PRICE_FROM_REF_PRICE, IS_OPTION, IS_ANNUAL_PRICES, ANNUAL_PRICE_MODE, IS_DIVIDE_QUARTAL_PRICES, IS_DOWN_PRICE, USE_MIN_RT_DA, min_RT, min_DA)
    
    timer.end('tp', "after calc price")
    
    # set value of current loop for choosen paths
    df_points['Step'] = 0
    df_points.Step = df_points.Step.astype(np.int16)
    
    df_points['iLoop'] = 0
    df_points.iLoop = df_points.iLoop.astype(np.int16)
    
    df_points['flagStop'] = 10
    df_points.flagStop = df_points.flagStop.astype(np.int8)
    
    # calc collateral
    df_points = calc_Collateral(df_points, CLUSTERS_iso, IS_UNITARY_COLLATERAL, CLUSTER_collateral, min_collateral_value, dict_class_mode_to_name[class_mode])    
    
    #if (CLUSTERS_iso == "SPP") & ((IS_UNITARY_COLLATERAL != 1) & (IS_UNITARY_COLLATERAL != True) & (IS_UNITARY_COLLATERAL != False)):
    #    df_points.loc[((df_points.Collateral_SPP > IS_UNITARY_COLLATERAL)), 'flagStop'] = 12
    
    # Fp_i0 = (Sum constr (inf*sum*possib)   -  price path)/ collateral
    df_points['Fp_i0'] = 0
    df_points['Fp_i03'] = 0
    df_points = calc_F_path0(df_points)
    df_points.Fp_i0 = df_points.Fp_i0.astype(np.float64)
    df_points.Fp_i03 = df_points.Fp_i03.astype(np.float64)
    
    #root_array_df = get_common_root(log_path,r'common\Array_df_FULL.csv')
    #df_points.to_csv(root_array_df)
        #calc F for Recent stage
    df_points['Fp_iR'] = df_points['WeekDA'] + df_points['WeekRT'] - df_points['Price']
    if IS_OFF_STAGE_R:
        min_recent_delta = 1000000
    else: min_recent_delta = 100
    coef_recent_neg_stageR = 0.1
    check_Fp_iR_for_stage_R = (df_points.Fp_iR >= min_recent_delta) & ((abs(df_points.minDART)) < coef_recent_neg_stageR*(df_points.WeekDA + df_points.WeekRT)) & ((df_points.WeekDA + df_points.WeekRT) > min_recent_delta)

    # throw out paths which are not meet the conditions :
    #   for  low price :  FV - price > min_price_delta
    #   for high price :   FV - price > min_price_fraction_delta* price
    low_price_goodF_condition = (min_price_fraction_delta*df_points.Price <= min_price_delta) & ((df_points.FV - df_points.Price) >= min_price_delta )
    high_price_goodF_condition = (min_price_fraction_delta*df_points.Price > min_price_delta) & ((df_points.FV - df_points.Price) > min_price_fraction_delta*df_points.Price )

    df_points_negF = df_points[~(low_price_goodF_condition | high_price_goodF_condition | check_Fp_iR_for_stage_R) ]#& (df_points.MWs==0)]    
    
    file_save_all_source_sink = os_path.join(log_path,'All_Path_Removed_From_RC_Tool.csv')
    df_points_negF.to_csv(file_save_all_source_sink, index=0, columns=['Source','Sink','Price','FV','Source1','Sink1'])
    # for further analisys use only positive F
    df_points = df_points[low_price_goodF_condition | high_price_goodF_condition | check_Fp_iR_for_stage_R | (df_points.MWs>0)]
    df_points.loc[(low_price_goodF_condition | high_price_goodF_condition | (df_points.MWs>0)),'flagStop'] = 0
    
    
    df_points['MWs'] = 0.0
    df_points.MWs = df_points.MWs.astype(np.float16)
    
    df_points['MWs_prev'] = 0.0
    df_points.MWs_prev = df_points.MWs_prev.astype(np.float16)
    
    df_points['Capital'] = (df_points.MWs*df_points.Price).astype(np.float32)
    
    df_points['Capital_t'] = (df_points.MWs*df_points.Price_t).astype(np.float32)
    df_points.loc[df_points.Collateral>=min_collateral_value,'Capital_Collateral'] = (df_points.MWs*df_points.Collateral).astype(np.float32)
       
    # set corr points, duplicated above, but for check
    df_points['Source1'] = df_points.Source.map(dict_rename)
    df_points.loc[df_points.Source1.isnull(),'Source1'] = df_points.Source
    df_points['Sink1'] = df_points.Sink.map(dict_rename)
    df_points.loc[df_points.Sink1.isnull(),'Sink1'] = df_points.Sink
    
    # write to disk paths and prices
    file_save_all_source_sink = os_path.join(log_path,'All_Path_Moved_From_RC_Tool.csv')
    df_points.to_csv(file_save_all_source_sink, index=0, columns=['Source','Sink','Price','FV','Source1','Sink1'])
     
    df_points.drop_duplicates(['Source1','Sink1'], inplace=True)
    df_points = df_points[df_points.Source1!=df_points.Sink1]
    timer.end('1','import points and prices   ')
           
    return (df_points, dict_rename, list_constr_for_FV, df_points_negF, min_recent_delta, coef_recent_neg_stageR)

def remove_restricted_path(CLUSTERS_iso, df_points, additional_data, log_path, path_to_fit_EESL, use_EESL, class_mode):
    
    # for CAISO additional data is df_points_sourcesink_source - to get path type from Type column - and choose only allowable path type
    if CLUSTERS_iso == 'CAISO':
        
        allowable_combinations_list = ['Generator+Scheduling Point','Generator+Load Aggregated Point','Generator+Trading Hub','Trading Hub+Scheduling Point','Trading Hub+Load Aggregated Point','Scheduling Point+Load Aggregated Point','Scheduling Point+Trading Hub'];
        
        #for CAISO additional data is df_points_sourcesink_source - to get path type from Type column
        dict_node_to_type = pd.Series(additional_data.Type.values,index=additional_data.Name).to_dict()
        df_points['PathType'] = df_points.Source.map(dict_node_to_type)+'+'+df_points.Sink.map(dict_node_to_type)
        df_points = df_points.loc[df_points.PathType.isin(allowable_combinations_list)]
    
    
    if (CLUSTERS_iso == 'SPP') & (use_EESL == 1):
        df_check_EESL = pd.read_csv(path_to_fit_EESL)
        dict_class_for_spp_EESL = {1: 'ON', 0: 'OFF'}
        df_check_EESL = df_check_EESL.loc[df_check_EESL.Class == dict_class_for_spp_EESL[class_mode]]
        
        dict_node_to_GroupID = pd.Series(df_check_EESL["Group ID"].values, index=(df_check_EESL["Settlement Location"])).to_dict()
        
        df_points['groupID_source'] = pd.to_numeric(df_points.Source.map(lambda x: dict_node_to_GroupID.get(x,9999))).round(3)
        df_points['groupID_sink'] = pd.to_numeric(df_points.Sink.map(lambda x: dict_node_to_GroupID.get(x,9119))).round(3)
        #df_points['groupID_source'] = df_points.groupID_source.astype(int)
        #df_points['groupID_sink'] = df_points.groupID_sink.astype(int)
        df_points=df_points.loc[df_points.groupID_source != df_points.groupID_sink]
    
    
    # for ERCOT additional data is (FTR lmp file, class_mode).  FTR lmp file - file with allowable pathes 
    # (in form "Source---Sink") and lmp for each class . if lmp ==  -111111 - means path is restricted for given class
    if CLUSTERS_iso == 'ERCOT':
        (FTR_lmp_file, class_mode) = additional_data
        df_ftr_lmp = pd.read_csv(FTR_lmp_file)
        # drop second line (which has 0 index since it is a first line after "header") as it is an additional header
        df_ftr_lmp.drop(0, axis=0, inplace=True)
        # [LMP    LMP.1    LMP.2]  headers are related to [PeakWE PeakWD Off-peak]
        # so choose only data with in needed column (according to class) value != -111111
        class_column = {1:'LMP.1', 0:'LMP.2', 2:'LMP'}[class_mode]
        df_ftr_lmp = df_ftr_lmp.loc[df_ftr_lmp[class_column].astype(float)!=-111111]
        df_points = df_points.loc[(df_points.Source+'---'+df_points.Sink).isin(df_ftr_lmp.Node)]
        
    return df_points

def fill_take_constr_columns(df_points, dict_points_influence, list_main_constraints, hack_take_best_source_sink_paths_max_count_nodes, min_influence_for_node, min_influence_for_path, dict_rename, log_file):
# rename if dipoles library
    list_val = [ k.split('+') + [v] for k, v in dict_points_influence.items()]
    df_inf = pd.DataFrame(list_val, columns = 'Node,ConstraintDevice,ContingencyDevices,Direction,Influence'.split(','))
    df_inf.Influence = df_inf.Influence.abs() 
    
    df_inf['Node1'] = df_inf.Node.map(lambda x: dict_rename.get(x,x))    
    # fix direction
    df_inf['Direction'] = pd.to_numeric(df_inf['Direction'].fillna(0), errors='ignore', downcast='integer')
    df_inf.Direction = df_inf.Direction.astype(str)
                                       
    df_inf['conID'] = df_inf.ConstraintDevice.fillna('') + '+' + df_inf.ContingencyDevices.fillna('') + '+' + df_inf.Direction
    df_inf.drop_duplicates(['Node1','conID'], inplace=True)
    list_rows_constraints_points = []
    for constr in list_main_constraints:
        #list_nodes_constr = df_inf[(df_inf.conID == constr) & (df_inf.Influence>=min_influence_for_node)].Node1.tolist()
        list_nodes_constr = df_inf[(df_inf.conID == constr) & (df_inf.Influence>=min_influence_for_node)].sort_values('Influence', ascending=False).Node1[:hack_take_best_source_sink_paths_max_count_nodes].tolist()
        
        list_nodes_constr = set(list_nodes_constr) # unique nodes
        list_nodes_constr = set(list(df_points.Sink1)+list(df_points.Source1))&list_nodes_constr
        list_nodes_constr = list(list_nodes_constr)
        # filter rows in dataframe with Source and sink in list_nodes_constr
        cond_found_paths_nodes = (df_points.Source1.isin(list_nodes_constr)) & (df_points.Sink1.isin(list_nodes_constr))
        cond_found_paths_one_node = (df_points.Source1.isin(list_nodes_constr)) | (df_points.Sink1.isin(list_nodes_constr))
        # filter rows with influence to constr > 0
        cond_found_paths_nodes_posit = cond_found_paths_nodes & (df_points[constr]>=min_influence_for_path*0.01)
        cond_found_paths_one_node_posit = cond_found_paths_one_node & (~cond_found_paths_nodes) & (df_points[constr]>=min_influence_for_path*0.01)
        cond_found_paths_nodes_non_zero_infl = (df_points[constr].abs()>=0.01)
        
        count_list_paths_all = df_points[cond_found_paths_nodes].shape[0]
        list_paths_all = ';'.join(['-'.join(x) for x in zip(df_points[cond_found_paths_nodes].Source,df_points[cond_found_paths_nodes].Sink,df_points[cond_found_paths_nodes][constr].astype(str))])
        
        count_list_paths_posit = df_points[cond_found_paths_nodes_posit].shape[0]
        list_paths_posit = ';'.join(['-'.join(x) for x in zip(df_points[cond_found_paths_nodes_posit].Source,df_points[cond_found_paths_nodes_posit].Sink,df_points[cond_found_paths_nodes_posit][constr].astype(str))])
        # renamed points in Source1 and Sink1 columns in df_points
        if (cond_found_paths_one_node_posit & cond_found_paths_nodes_posit).any(): 
            raise ValueError('intersected take constr and by one node')
        df_points.loc[cond_found_paths_nodes_posit, 'Take_constraints'] += constr + '('+(df_points[constr]*100).round().astype(str)+'%);'
        df_points.loc[cond_found_paths_one_node_posit, 'Take_constraints_byOneNode'] += constr+'('+(df_points[constr]*100).round().astype(str)+'%);'
        df_points.loc[cond_found_paths_nodes_non_zero_infl,'is_non_zero_infl'] = 1
        list_rows_constraints_points.append((constr,len(list_nodes_constr),';'.join(list_nodes_constr),count_list_paths_all,list_paths_all,count_list_paths_posit,list_paths_posit))
    
    # write to file Constr,uniq points
    # with list of paths
    #'constraintID','count_points','list_points','count_paths_all','list_paths_all','count_paths_posit','list_paths_posit'
    pd.DataFrame(list_rows_constraints_points, columns=['constraintID','count_points','list_points','count_paths_all','list_paths_all','count_paths_posit','list_paths_posit']).to_csv(log_file, index=0, columns=['constraintID','count_points','count_paths_all','count_paths_posit'])
    
    return df_points


def path_optimizer_main_loop(CLUSTERS_iso, iLoop, df_points, log_path, save2_columns, stage, list_main_constraints, constraints_taken_by_limits, dict_constr_FV, IS_CHOOSE_CONSTR_BY_FV, n_main_constr, dict_coeff_constraints, dict_coeff_constraints_from_NN_table, dict_coef_portfolio, loop_count_RM, MWs_step, dict_coef_portfolio_neg, IS_WRITE_NEG_CONSTR_COMMENT, mws_max_path, loop_count_Fcalc, dict_coeff_constraints_orig_table, calc_F_main,list_constr_for_FV, mws_max_node, loop_count_takePath, list_column_order, minChange, list_All_constraints, list_All_constraints_from_NN_table, cur_Nplus_coef, n_plus_step,  max_N_plus, df_points_take_constraints_by_one_node,df_points_no_take_constraints, df_points_bad_price, dict_coef_portfolio_prev, CLUSTER_F_correction_coef, min_price_delta, DO_FULL_OUTPUT_FILES, list_All_mws_constraints, dict_constr_in_dipoles, dict_main_constraints, class_mode, CLUSTERS_constr_table, min_price_fraction_delta, coef_recent_neg, min_recent_delta, coef_recent_neg_stageR, IS_CHECK_MinDART_RECENT):
    
    print()
    print('Loop {}: start loop cycle'.format(iLoop))
    print('Shape of df_points: {}'.format(df_points.shape[0]))
    
    LogStageValue('Loop {}: start loop cycle'.format(iLoop), log_path, "id")
    LogStageValue('Loop {}: shape of df_points: {}'.format(iLoop,df_points.shape[0]), log_path, "id")

    timer.start('loop')
    timer.start('2')
    timer.start('1')
        
    # remove all paths which have stop flag
    if df_points[(df_points.flagStop>0) & (df_points.flagStop != 10) & (df_points.MWs==0)].shape[0]>0:
        print('inside if')
        before_rows = df_points.shape[0]
        df_points[(df_points.flagStop>0) & (df_points.flagStop != 10) & (df_points.MWs==0)].to_csv(get_common_root(log_path,r'save2\{}_{}_a_flagStop_not_zero.csv'.format(iLoop,"id")),index=0,columns=save2_columns, float_format='%g')
        # save paths if flagStop < 2  and with MW>0
        print('before reindex')
        df_points = df_points[(df_points.flagStop<2) | (df_points.MWs>0) | (df_points.flagStop == 10)].reindex()
        after_rows = df_points.shape[0]
        print('Flagstop>0 Rows before: {}, Rows after: {}'.format(before_rows,after_rows))
        LogStageValue('Loop {}: Flagstop>0 Rows before: {}, Rows after: {}'.format(iLoop,before_rows,after_rows), log_path, "id")
        #df_points.loc[df_points.flagStop != 1, 'flagStop'] = 0
    print()    
    timer.end('2', 'loop init')
    
    df_points.iLoop = iLoop
        
    df_points.MWs_prev = df_points.MWs
       
    # choose paths for given constraint
    if (stage == 1) | (stage == 2):
        
        untaken_main_constr = list(set(list_main_constraints)-constraints_taken_by_limits)
# =============================================================================
#         untaken_main_constr_df = pd.DataFrame(untaken_main_constr)
# # =============================================================================
# #         root_untake_constr = get_common_root(log_path,r'common\UnTakeConstr_{}.csv'.format(iLoop))
# #         untaken_main_constr_df.to_csv(root_untake_constr)
# # =============================================================================
# =============================================================================
        LogStageValue('Loop {}: Untaken constraints N: {}, constr limit exs N: {}'.format(iLoop,len(untaken_main_constr),len(constraints_taken_by_limits)), log_path, "id")
        
        curr_constr_main = choose_cur_main_constr(df_points, list_main_constraints, untaken_main_constr, dict_constr_FV, IS_CHOOSE_CONSTR_BY_FV)
        
        if stage == 1 :
            df_points.loc[(~(df_points.Take_constraints.str.contains(curr_constr_main, regex=False))) & (df_points.flagStop==0),'flagStop']=-9
            shape_with_curr_constraint = df_points[df_points.Take_constraints.str.contains(curr_constr_main, regex=False, na = False)].shape[0]
        if stage == 2:
            df_points.loc[(~(df_points.Take_constraints_byOneNode.str.contains(curr_constr_main, regex=False))) & (df_points.flagStop==0),'flagStop']=-9
            shape_with_curr_constraint = df_points[df_points.Take_constraints_byOneNode.str.contains(curr_constr_main, regex=False, na = False)].shape[0]
        print('')
        print('Current index: {}/{}'.format(len(constraints_taken_by_limits)+1, n_main_constr))
        print('Current constr: {}'.format(curr_constr_main))
        print('Current coef of constr: {}/{}'.format(df_points['mws_'+curr_constr_main].sum(),dict_coeff_constraints[curr_constr_main][0]))
        print('Shape with curr constr: {}'.format(shape_with_curr_constraint))
        print('')        
        
        LogStageValue('Loop {}: choose constraint\n\tCurrent index: {}/{} Current constr: {} Current coef of constr: {}/{} Shape with curr constr: {} Constr FV: {}'.format(iLoop,len(constraints_taken_by_limits)+1,n_main_constr,curr_constr_main,df_points['mws_'+curr_constr_main].sum(),dict_coeff_constraints[curr_constr_main][0],df_points.loc[df_points.Take_constraints.str.contains(curr_constr_main, regex=False)].shape[0], dict_constr_FV[curr_constr_main]), log_path, "id")
    timer.end('1', 'choose constraint')
    # ---------- RISK managment for paths-------------
        
    timer.start('21')
            
    # check if N+ for portfolio is already more than needed N+. if so - do not calc RM
    # dict_coeff_constraints - [[N-, N+], sum, possib, sum*possib]
    is_curr_constr_Np_no_excess = True
    is_present_paths_with_curr_constraint = True
    if (stage == 1) | (stage == 2):
        is_curr_constr_Np_no_excess = dict_coef_portfolio[curr_constr_main] <= dict_coeff_constraints[curr_constr_main][0][1]
        is_present_paths_with_curr_constraint = (shape_with_curr_constraint != 0)
    
    if not is_curr_constr_Np_no_excess:
        LogStageValue('Loop {}: empty RM - N+ exceed '.format(iLoop), log_path, "id")
        
    if not is_present_paths_with_curr_constraint:
        LogStageValue('Loop {}: empty RM - No paths with curr constraint '.format(iLoop), log_path, "id")
    
    if is_curr_constr_Np_no_excess & is_present_paths_with_curr_constraint:
        loop_count_RM += 1
        df_points = check_Nminus_violation(dict_coeff_constraints.keys(), df_points, dict_coef_portfolio_neg, dict_coeff_constraints, MWs_step, iLoop, IS_WRITE_NEG_CONSTR_COMMENT)
        df_points.loc[~df_points.isNminus_ok, 'flagStop'] = 3
    timer.end('21', 'loop check coef_portfolio and set flagStop to limit constraints')
         
    #================== RM end ==============  
        
    
#==================  Caclulate function for each path ==============
    timer.start('2')
    # check if there is any path with good flagstop (==0)
    is_any_path_present_by_flagStop = ((df_points.flagStop==0) & (df_points.MWs<mws_max_path)).any()
    
    if not (is_any_path_present_by_flagStop &  is_curr_constr_Np_no_excess & is_present_paths_with_curr_constraint):
        LogStageValue('Loop {}: empty flagstop == 0 / N+ exceed / no paths for curr constraint. Before calc F'.format(iLoop), log_path, "id")
        
    # if there are no good paths or N+ for portfolio is already more than needed N+  - dont calc F
    if is_any_path_present_by_flagStop & is_curr_constr_Np_no_excess & is_present_paths_with_curr_constraint:   
        loop_count_Fcalc += 1     
        
        fix_F_for_Nplus_excess(df_points, dict_coeff_constraints_orig_table, dict_coef_portfolio, MWs_step, calc_F_main,list_constr_for_FV, list_main_constraints)
        #block paths with negative delta F
        df_points.loc[((df_points.Fp_i <= 0) & (df_points.flagStop == 0)), 'flagStop'] = 7
                       
    else:
        df_points.loc[(df_points.flagStop == 0), 'flagStop'] = -4
    timer.end('2', 'loop calc for each row new Fp_i')
#================== end calc F end ==============        


#============ set flag stop if (FV-Price)*coef_recent_neg < |minDART| ===
    if IS_CHECK_MinDART_RECENT:
        df_points.loc[(((df_points.Fp_i0)*(df_points.Collateral)*coef_recent_neg)<(df_points.minDART.abs())), 'flagStop'] = 5
#===== end of set flag stop if (FV-Price)*coef_recent_neg < |minDART| ===
    

#============ set flag stop if exceed mws ===
    df_points.loc[(df_points.MWs > mws_max_path) & (df_points.Stage !='start portf'),'MWs'] = mws_max_path
    df_points.loc[(df_points.MWs == mws_max_path) & (df_points.Stage !='start portf'),'flagStop'] = 1
    # also remove paths with source or sink with mw = mws_max_node
    gr_by_source = df_points.groupby('Source', sort=False).agg({'MWs':sum}).reset_index()
    sorces_exceed_mws  = gr_by_source[gr_by_source.MWs >= mws_max_node].Source
    
    gr_by_sink = df_points.groupby('Sink', sort=False).agg({'MWs':sum}).reset_index()
    sinks_exceed_mws  = gr_by_sink[gr_by_sink.MWs >= mws_max_node].Sink

    df_points.loc[(df_points.Source.isin(sorces_exceed_mws)) | (df_points.Sink.isin(sinks_exceed_mws)), 'flagStop'] = 4
#============ end of set flag stop if exceed mws ===


#================= 10 best paths start ============    
    timer.start('2')
    #get only paths that have no flagStop, have funciton >0
    cond_not_limitMWs = (df_points.flagStop==0) & (df_points.MWs<mws_max_path) # & (df_points.Fp_i>0.0)
    
    LogStageValue('Loop {}: available paths exists: {} shape {} all: {}'.format(iLoop,cond_not_limitMWs.any(),cond_not_limitMWs[cond_not_limitMWs == True].shape[0],cond_not_limitMWs.shape[0]), log_path, "id")
            
    if stage != "R":
        # take 10 best paths by FV
        if 'Fp_i' not in list_column_order:
            list_column_order.append('Fp_i')
            
        if 'Fp_i' not in df_points.columns:
            df_points['Fp_i'] = df_points.Fp_i0
        df_func = df_points[cond_not_limitMWs].nlargest(10, 'Fp_i')
        if df_func.shape[0]>0:
            loop_count_takePath += 1
            file_Pf_i = get_common_root(log_path,r'common\Fp_i_{}.csv'.format(iLoop))
            df_func.to_csv(file_Pf_i,columns = list_column_order, float_format='%g')
            ReadOnly(file_Pf_i)
            # gen only minChange (default 1) largest paths
            min_value_weight = df_func['Weight'].min()
            df_func_weight = df_func.loc[df_func.Weight == min_value_weight]
            final = df_func_weight.nlargest(minChange, 'Fp_i')
                    
            # if path is in main cluster simply add MWs_Step to column MWs of paths
            if df_points[(df_points.Source == final.Source.iloc[0]) & (df_points.Sink == final.Sink.iloc[0])].shape[0]>0:
                LogStageValue('Loop {}: MAIN CLUSTER: Add {} to path: {}-{}'.format(iLoop,MWs_step,final.Source.iloc[0],final.Sink.iloc[0]), log_path, "id")
                df_points.loc[(df_points.Source == final.Source.iloc[0]) & (df_points.Sink == final.Sink.iloc[0]), 'MWs'] = df_points.MWs+MWs_step
                df_points.loc[(df_points.Source == final.Source.iloc[0]) & (df_points.Sink == final.Sink.iloc[0]) & (df_points.Step<1), 'Step'] = iLoop
                df_points.loc[(df_points.Source == final.Source.iloc[0]) & (df_points.Sink == final.Sink.iloc[0]), 'Stage'] = df_points.Stage+'stage '+str(stage)+';'
        
        else: # if file fp_i.csv not generated
            LogStageValue('Loop {}: no file gen Fp_i'.format(iLoop), log_path, "id")
             
            if (stage == 1) | (stage == 2):
                # if loop_all_constraints_by_limits_taken = main constr count, increment pos limits of constraints N+
                constraints_taken_by_limits.add(curr_constr_main)
        final = 0
        df_func_weight = 0
        min_value_weight = 0
    
    elif stage == "R":
        # take 10 best paths by Recent
        check_Fp_iR_for_stage_R2 = (df_points.Fp_iR > min_recent_delta) & ((abs(df_points.minDART)) < coef_recent_neg_stageR*(df_points.WeekDA + df_points.WeekRT)) & ((df_points.WeekDA + df_points.WeekRT) > min_recent_delta)
        cond_not_R_limitMWs2 = check_Fp_iR_for_stage_R2 & ((df_points.flagStop==0) | (df_points.flagStop==10)) & (df_points.MWs<mws_max_path)
        df_funcR = df_points[cond_not_R_limitMWs2].nlargest(10, 'Fp_iR')
        if df_funcR.shape[0]>0:
            loop_count_takePath += 1
            file_Pf_iR = get_common_root(log_path,r'common\Fp_iR_{}.csv'.format(iLoop))
            df_funcR.to_csv(file_Pf_iR,columns = list_column_order, float_format='%g')
            ReadOnly(file_Pf_iR)
            # gen only minChange (default 1) largest paths
            final_R = df_funcR.nlargest(minChange, 'Fp_iR')
                    
            # if path is in main cluster simply add MWs_Step to column MWs of paths
            if df_points[(df_points.Source == final_R.Source.iloc[0]) & (df_points.Sink == final_R.Sink.iloc[0])].shape[0]>0:
                LogStageValue('Loop {}: MAIN CLUSTER: Add {} to path: {}-{}'.format(iLoop,MWs_step,final_R.Source.iloc[0],final_R.Sink.iloc[0]), log_path, "id")
                df_points.loc[(df_points.Source == final_R.Source.iloc[0]) & (df_points.Sink == final_R.Sink.iloc[0]), 'MWs'] = df_points.MWs+MWs_step
                df_points.loc[(df_points.Source == final_R.Source.iloc[0]) & (df_points.Sink == final_R.Sink.iloc[0]) & (df_points.Step<1), 'Step'] = iLoop
                df_points.loc[(df_points.Source == final_R.Source.iloc[0]) & (df_points.Sink == final_R.Sink.iloc[0]), 'Stage'] = df_points.Stage+'stage '+str(stage)+';'
    
        else: # if file fp_i.csv not generated
            LogStageValue('Loop {}: no file gen Fp_iR'.format(iLoop), log_path, "id")
            df_points.loc[df_points.flagStop == 0,'flagStop'] = -10
    final_R = 0
    # update MWs_constr ,Capital, collateral, mws 
    
    
    for constr in list_All_constraints:
        if constr in df_points.columns:
            df_points['mws_'+constr] = df_points[constr]*df_points.MWs

    df_points.Capital = df_points.MWs*df_points.Price.astype(np.float64)
    
    df_points.Capital_t = (df_points.MWs*df_points.Price).astype(np.float32)
    df_points.Capital_Collateral = (df_points.MWs*df_points.Collateral_t).astype(np.float32)
    df_points.loc[df_points.Capital_Collateral<0,'Capital_Collateral']=0 # make negative collat = 0
    
    # calculate coeff for portfolio
    (dict_coef_portfolio, dict_coef_portfolio_neg) = calc_RM_coef_portfolio(dict_coeff_constraints.keys(), df_points)
    
    timer.end('2', 'get best path')
    
# =============================================================================
#     hack_limit_pos_step_by_step implementation
#     check cond and change N+ limits
# =============================================================================
    timer.start('2')
    if (stage == 1) | (stage == 2):
        df_points.loc[df_points.flagStop == -9, 'flagStop'] = 0
     
    (df_points, dict_coef_portfolio_prev, cond_good_df_points, constraints_taken_by_limits, stage, calc_F_main, cur_Nplus_coef, dict_coeff_constraints) = finalize_loop(df_points, mws_max_path, constraints_taken_by_limits, n_main_constr, cur_Nplus_coef, n_plus_step,  max_N_plus, stage, calc_F_main, df_points_take_constraints_by_one_node,df_points_no_take_constraints, df_points_bad_price, dict_coeff_constraints, dict_coeff_constraints_orig_table, dict_coef_portfolio, dict_coef_portfolio_prev, list_constr_for_FV, list_main_constraints, CLUSTER_F_correction_coef, iLoop, min_price_delta,min_price_fraction_delta, log_path)
    timer.end('2','finalize loop ')
# ==================== write output part ===========================================
    print('select portfolio')    
    df_portfolio_final = df_points[df_points.MWs>0].copy()
    
    if not (is_any_path_present_by_flagStop &  is_curr_constr_Np_no_excess & is_present_paths_with_curr_constraint):
        LogStageValue('Loop {}: empty flagstop == 0 / N+ exceed / no paths for curr constraint. Before write output'.format(iLoop), log_path, "id")
                  
    if is_any_path_present_by_flagStop & is_curr_constr_Np_no_excess & is_present_paths_with_curr_constraint & (DO_FULL_OUTPUT_FILES | (iLoop%50 == 0)):
        print('write portfolio')   
        # write portfolio
        write_portfolio_file(get_common_root(log_path,r'common'), iLoop, df_portfolio_final, list_column_order)
        
        # write constraints
        timer.start('2')
        print('write constraints')   
        write_constraints_file(get_common_root(log_path,r'common'), iLoop, df_portfolio_final, list_All_constraints_from_NN_table, list_All_mws_constraints, dict_constr_in_dipoles, dict_coef_portfolio, dict_coef_portfolio_neg, dict_main_constraints, dict_coeff_constraints_from_NN_table)
        timer.end('2','save constraints coef ')
    
#================== write output end ==============
    if iLoop == 1:
        root_array_df = get_common_root(log_path,r'common\Array_df_after_first_loop{}.csv'.format(iLoop))
        df_points.to_csv(root_array_df)
    
  
    print('sum MWs: ',df_portfolio_final.MWs.sum())
    print('Capital: ',df_portfolio_final.Capital.sum())
    print('Capital Collateral: ',df_portfolio_final.Capital_Collateral.round(2).sum())

    print()
    print("-------------")
    print('iLoop: ',iLoop)
    
    t_end = timer.end('loop')
    LogStageValue('Loop {}: loop count RM = {} ,  loop count F calc = {}, loop count Take Path = {}'.format(iLoop,loop_count_RM, loop_count_Fcalc, loop_count_takePath), log_path, "id")
    LogStageValue('Loop {}: end in {} sec'.format(iLoop,t_end), log_path, "id")
    print("-------------")
    
    
# =============================================================================
#     Check if no paths available 
# =============================================================================
    if not cond_good_df_points.any():
        
        df_points[(df_points.flagStop>0) & (df_points.MWs==0)].to_csv(get_common_root(log_path,r'save2\{}_{}_c_no_paths_available.csv'.format(iLoop,"id")),index=0,columns=save2_columns, float_format='%g')
        timer.end('main','End of script')
        print('CLUSTER_PATH_ROOT: ', log_path)
        LogStageValue("that's all folks", log_path, "id")
        print("that's all folks")
        write_portfolio_file(get_common_root(log_path,r'common'), iLoop+1, df_portfolio_final, list_column_order)
        write_constraints_file(get_common_root(log_path,r'common'), iLoop+1, df_portfolio_final, list_All_constraints_from_NN_table, list_All_mws_constraints, dict_constr_in_dipoles, dict_coef_portfolio, dict_coef_portfolio_neg, dict_main_constraints, dict_coeff_constraints_from_NN_table)
        bid_class = dict_class_mode_to_name[class_mode]
            
        create_final_result_files(CLUSTERS_iso, os_path.join(get_common_root(log_path,r'common'),'portfolio_{}.csv'.format(iLoop+1)), os_path.join(get_common_root(log_path,r'common'),'constraints_{}.csv'.format(iLoop+1)) , CLUSTERS_constr_table, bid_class, class_mode)
        return 0 #raise ValueError("that's all folks")   
    
    return (df_points, stage, constraints_taken_by_limits, dict_coeff_constraints, dict_coef_portfolio, loop_count_RM, dict_coef_portfolio_neg, loop_count_Fcalc, calc_F_main, loop_count_takePath, cur_Nplus_coef, dict_coef_portfolio_prev, coef_recent_neg, IS_CHECK_MinDART_RECENT)