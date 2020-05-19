# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:11:44 2019

@author: Администратор
"""
import sys
import pandas as pd
from os import path as os_path, listdir as os_listdir, remove as os_remove
import numpy as np
from datetime import datetime

folderSnippets = r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\snippets'
sys.path.insert(0,folderSnippets)
from uFunc import PrintOneRow
folderSnippets = r'z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\standard_functions'
sys.path.insert(0,folderSnippets)
from market_standard_functions import fix_constr_conting_dir

def get_influence_library_to_dict_with_additional_lib(iso, inf_lib_path_main, inf_lib_path_additional, nodes_from_additional, constraints_to_check, output_log_path, is_clear_lib_inf_cash=False, is_return_second_dict = False):
    
    is_return_df=False
    
    # there is allways should be something in inf_lib_path_main. raise error if not
    if (not os_path.isfile(inf_lib_path_main)) & (not os_path.isdir(inf_lib_path_main)):
        raise ValueError('There is no ', inf_lib_path_main,'!!! Please check it!!!')
    
    # if there is no inf_lib_path_additional, just normally load inf_lib_path_main
    if (not os_path.isfile(inf_lib_path_additional)) & (not os_path.isdir(inf_lib_path_additional)):
        print('Warning!!! no additional lib - just load from ', inf_lib_path_main)
        return get_influence_library_to_dict(iso, inf_lib_path_main, output_log_path, is_clear_lib_inf_cash, is_return_df, is_return_second_dict)
    
    is_return_second_dict = False
    dict_points_influence_main = get_influence_library_to_dict(iso, inf_lib_path_main, output_log_path, is_clear_lib_inf_cash, is_return_df, is_return_second_dict)
    
    if len(nodes_from_additional) == 0:
        return dict_points_influence_main
    
    # get all needed node-constraint pairs in form of  inf lib key = Node+Constraint+Contingency+Direction
    nodes_consr_pairs_to_delete_from_main_dict = [node+'+'+constraint for node in nodes_from_additional for constraint in constraints_to_check]
    
    # remove keys which will be taken from additional lib
    [dict_points_influence_main.pop(key, None) for key in nodes_consr_pairs_to_delete_from_main_dict]
    
    dict_points_influence_additional_full = get_influence_library_to_dict(iso, inf_lib_path_additional, output_log_path, is_clear_lib_inf_cash, is_return_df, is_return_second_dict)
    
    dict_points_influence_additional = {key:dict_points_influence_additional_full.get(key) for key in nodes_consr_pairs_to_delete_from_main_dict if dict_points_influence_additional_full.get(key,None)!=None}
    
    # add additional needed dict to main one
    # use dict_points_influence_additional for updating since d1.update(d2) func replace all existing key in firts dict (d1) to values from second (d2)
    dict_points_influence_additional.update(dict_points_influence_main)
    
    return dict_points_influence_additional
    
# dipoles influences library
# return dict: key = Node+Constraint+Contingency+Direction, value = Influence
def get_influence_library_to_dict(iso, inf_lib_path, output_log_path, is_clear_lib_inf_cash=False, is_return_df=False, is_return_second_dict = False):
        
    temp_inf_file = os_path.join(inf_lib_path,'inf_dict.npy')
    temp_inf_file_2 = os_path.join(inf_lib_path,'inf_dict_2.npy')
    if is_return_df:
        temp_inf_file = os_path.join(inf_lib_path,'inf_df.pkl')
        temp_inf_file_2  = os_path.join(inf_lib_path,'inf_df.pkl')
    today = datetime.today().strftime('%Y-%m-%d')
    
    if (os_path.isfile(temp_inf_file)) & (os_path.isfile(temp_inf_file_2)):
        if (not is_clear_lib_inf_cash) & (datetime.fromtimestamp(os_path.getmtime(temp_inf_file)).strftime('%Y-%m-%d')==today) :
            if is_return_df:
                return pd.read_pickle(temp_inf_file)
            elif is_return_second_dict:
                return np.load(temp_inf_file_2, allow_pickle=True).item()
            else:
                return np.load(temp_inf_file, allow_pickle=True).item()
    
    if os_path.isfile(temp_inf_file):
        os_remove(temp_inf_file)
    
    if os_path.isfile(temp_inf_file_2):
        os_remove(temp_inf_file_2)
    
    if os_path.isfile(inf_lib_path):    
        df_lib_corr_all = pd.read_csv(inf_lib_path)
    elif  os_path.isdir(inf_lib_path):
        filenames = os_listdir(inf_lib_path)
        filenames = list(map(lambda f: os_path.join(inf_lib_path,f),filenames)) # add folder
        dfs = []
        list_constr_comments = []
        for filename in filenames:
            if os_path.isfile(filename) & ('.csv' in filename):
                df_tmp = pd.read_csv(filename)
                df_tmp.rename(columns={'Contingency':'ContingencyDevices'},inplace=True)
                # check if file has needed columns
                if ('Influence' not in list(df_tmp.columns)) | ('Node' not in list(df_tmp.columns)):
                    list_constr_comments.append((filename, os_path.basename(filename),'File is not dipole'))
                    continue
                
                if (df_tmp.shape[0] == 0):
                    list_constr_comments.append((filename, os_path.basename(filename),'File is null'))
                    continue
                
                if (iso == "ERCOT")  | (iso == "NYISO") | (iso == "ISONE"):
                    df_tmp.Direction = 1
                
                df_tmp = fix_constr_conting_dir(iso, df_tmp)
                
                df_tmp['Key'] = df_tmp.ConstraintDevice.fillna('') + '+' + df_tmp.ContingencyDevices.fillna('') + '+' + df_tmp.Direction
                df_tmp_c=df_tmp.copy()
                
                df_tmp = df_tmp[~df_tmp.Influence.astype(str).str.contains('Abs')]
                df_tmp = df_tmp[~df_tmp.Influence.astype(str).str.contains('not')]
                df_tmp = df_tmp[df_tmp.Influence.notnull()]
                df_tmp.Influence = df_tmp.Influence.astype(np.float64)
                
                # check if all data was without influences
                if df_tmp.shape[0] == 0:
                    print()
                    print("throwing constr inf file {} because there were no number inf in file".format(os_path.basename(filename)))
                    list_constr_comments.append((filename, df_tmp_c.Key.iloc[0],'no number infl in file'))
                    continue
                
                 # check if file has correct influence (if all ( > 70%) inf > 80 mean there is something wrong)
                if (df_tmp[df_tmp.Influence>80].shape[0]/df_tmp.shape[0]>0.7) & (iso != "ISONE"):
                    print()
                    print("throwing constr inf file {} because more than half inf > 80 ({}/{})".format(os_path.basename(filename),df_tmp[df_tmp.Influence>80].shape[0],df_tmp.shape[0]))
                    list_constr_comments.append((filename, df_tmp.Key.iloc[0],'more than 70% inf > 80'))
                    continue
                 # check if file has correct influence (if inf > 100 mean there is something wrong)
                #if (df_tmp.Influence>101).any():
                    #print()
                    #print("throwing constr inf file {} because one of more inf > 100".format(os_path.basename(filename)))
                    #list_constr_comments.append((filename, df_tmp.Key.iloc[0],'one of more inf > 100'))
                    #continue
                 # check if file has correct influence (if all (100%!) inf < 1% mean there is something wrong)
                if df_tmp[df_tmp.Influence<1].shape[0]/df_tmp.shape[0]>=1.0:
                    print()
                    print("throwing constr inf file {} because all inf < 1 ({}/{})".format(os_path.basename(filename),df_tmp[df_tmp.Influence<1].shape[0],df_tmp.shape[0]))
                    list_constr_comments.append((filename, df_tmp.Key.iloc[0],'all inf < 1'))
                    continue
                                                    
                dfs.append(df_tmp)     
                
                PrintOneRow("{}: {}/{} {}%".format(os_path.basename(filename),filenames.index(filename),len(filenames),round(filenames.index(filename)/len(filenames)*100)).ljust(200))
        print()        
        df_lib_corr_all = pd.concat(dfs, ignore_index=True)  # Concatenate all data into one DataFrame
   
        pd.DataFrame(list_constr_comments, columns=['File', 'Key','Comment']).to_csv(os_path.join(output_log_path,'Dipoles_error.csv'), index=0)
    else:
        raise ValueError('Unknown structure for influence lib: ',inf_lib_path, ' please check it!')

    df_lib_corr_all.rename(columns={'Contingency':'ContingencyDevices'},inplace=True)
    # add minus to Source
    assert(df_lib_corr_all[(df_lib_corr_all.Influence<0)].shape[0] == 0), 'in dipoles Influences < 0'
    df_lib_corr_all.loc[(df_lib_corr_all.Type == 'Source') & (df_lib_corr_all.Influence >= 0), 'Influence'] = df_lib_corr_all['Influence']*-1
    assert(df_lib_corr_all[((df_lib_corr_all.Type == 'Source') & (df_lib_corr_all.Influence > 0))].shape[0] == 0), 'Source Influence  > 0'
    assert(df_lib_corr_all[((df_lib_corr_all.Type == 'Sink') & (df_lib_corr_all.Influence < 0))].shape[0] == 0), 'Sink Influence < 0'
    
    #df_lib_corr_all['OnlyZero'] = df_lib_corr_all.groupby(['ConstraintDevice','Node'])['MedianInfluence'].transform(np.median)
    # group and get median (not needed for dipoles, but why not)
    df_lib_corr_group_base = df_lib_corr_all.groupby(['ConstraintDevice','Direction','Node']).agg({'Influence':['median']})
    df_lib_corr_group_base.columns = ['_'.join(x) if len(x)>=2 else x for x in df_lib_corr_group_base.columns.ravel()]
    df_lib_corr_group_base.reset_index(inplace=True)
    df_lib_corr_group_base['ContingencyDevices'] = ''
    # with contingency
    df_lib_corr_group = df_lib_corr_all.groupby(['ConstraintDevice','ContingencyDevices','Direction','Node']).agg({'Influence':['median']})
    df_lib_corr_group.columns = ['_'.join(x) if len(x)>=2 else x for x in df_lib_corr_group.columns.ravel()]
    df_lib_corr_group.reset_index(inplace=True)

    # merge constraints base and with contingnency
    df_lib_corr_group_all = pd.concat([df_lib_corr_group,df_lib_corr_group_base],ignore_index=True)
    # fix direction
    df_lib_corr_group_all['Direction'] = pd.to_numeric(df_lib_corr_group_all['Direction'].fillna(0), errors='ignore', downcast='integer')
    df_lib_corr_group_all.Direction = df_lib_corr_group_all.Direction.astype(str)
    
    df_lib_corr_group_all['MedianInfluenceEx'] = df_lib_corr_group_all.Influence_median.round(2)
    df_lib_corr_group_all.MedianInfluenceEx = df_lib_corr_group_all.Influence_median.astype(np.float32) 
    
    # save also errors file
    temp_err_file = os_path.join(inf_lib_path,'inf_err.txt')
    pd.DataFrame(list_constr_comments, columns=['File', 'Key','Comment']).to_csv(temp_err_file, index=0)
    print('save err data to ', temp_err_file)
    
    if is_return_df:
        df_lib_corr_group_all[['Node', 'ConstraintDevice','ContingencyDevices','Direction','MedianInfluenceEx']].to_pickle(temp_inf_file)
        return df_lib_corr_group_all[['Node', 'ConstraintDevice','ContingencyDevices','Direction', 'MedianInfluenceEx']]
    
    # add unique key Node,Constraint,Contingency,Direction
    df_lib_corr_group_all["ID_const"] = df_lib_corr_group_all.ConstraintDevice.fillna('') + '+' + df_lib_corr_group_all.ContingencyDevices.fillna('') + '+' + df_lib_corr_group_all.Direction
    df_lib_corr_group_ID_const = df_lib_corr_group_all[['Node','ID_const','MedianInfluenceEx']].copy()
    df_lib_corr_pivot = pd.pivot_table(df_lib_corr_group_ID_const, values='MedianInfluenceEx', index=['Node'],columns=['ID_const'], fill_value=0)
    #df_lib_corr_pivot = df_lib_corr_group_ID_const.pivot(index='Node', columns='ID_const')['MedianInfluenceEx']
    dict_points_influence_ID_const = df_lib_corr_pivot.to_dict()
    np.save(temp_inf_file_2, dict_points_influence_ID_const)
        
    df_lib_corr_group_all['NC'] = df_lib_corr_group_all.Node+ '+' + df_lib_corr_group_all.ConstraintDevice.fillna('') + '+' + df_lib_corr_group_all.ContingencyDevices.fillna('') + '+' + df_lib_corr_group_all.Direction

    # return dict, key = Node+Constraint+Contingency+Direction, value = Influence
    dict_points_influence = pd.Series(df_lib_corr_group_all.MedianInfluenceEx.values,index=df_lib_corr_group_all.NC).to_dict()
    np.save(temp_inf_file, dict_points_influence)
    
    
    if is_return_second_dict == True:
        return dict_points_influence_ID_const
    
    else: return dict_points_influence
    
    
