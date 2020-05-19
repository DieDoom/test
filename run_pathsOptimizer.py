import sys
sys.path.insert(0, r"z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\pathsOptimizer")
sys.path.insert(0, r"z:\Dropbox (PowerMarketsUR)\Moscow Office\Software\Py\snippets")
from pathsOptimizer_main import run_path_optimizer
from fileIO import CheckToolVersion

# ---------- MAIN parameters ----------

OUTPUT_PATH = r"Z:\tmp\1182\MISO\off2" # output folder

CLUSTERS_iso = 'MISO'

# class_mode = 0 - off-peak
# class_mode = 1 - peak
# class_mode = 2 - peak we
class_mode = 1

# table with constraints MonthSum, Coeff limits for portfolio for peak and off-peak
CLUSTERS_constr_table = r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\MISO\Bids\2020_04\Study\StrongConstr\MISO_Constraints_Monthly_Sum_MainStrong_Short_2020-04-01__2020-04-30_NEW2.csv"

# fill prediction price file OR shadow price file. If from predicted - set flag IS_CALC_PRICE_FROM_MEAN_SIGMA to True
IS_CALC_PRICE_FROM_MEAN_SIGMA = True # if true - use CLUSTERS_price_distribution_file
CLUSTERS_price_distribution_file = r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\MISO\Bids\2020_04\Autopath\Pred_Apr.csv"
CLUSTERS_shadow_price = r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\ERCOT\Bids\2019_10\Authopas\Common_SourceAndSinkShadowPrices_2019.SEP.Monthly.Auction_AUCTION.csv" # price for each tradable point for peak and off-peak
# !!!  2 parameters below works only if IS_CALC_PRICE_FROM_MEAN_SIGMA == True !!!
# if IS_DOWN_PRICE = true - set predicted price as maximum between (real pred price - coef_sigma_down*sigma_minus) and (3minFTR column value)
IS_DOWN_PRICE = False
coef_sigma_down = 1.5 
ignor_3minFTR_for_down_price = False #do True if use BIG coef_sigma_down ( > 3 )
#if true - no take paths with DA|RTsumm30days < min_RT|min_DA
USE_MIN_RT_DA = False
min_RT = 0
min_DA = 0

#if true - no take paths with (FV-Price)*coef_recent_neg < |minDART|
IS_CHECK_MinDART_RECENT = True
coef_recent_neg = 0.1

#if True stage R no calc
IS_OFF_STAGE_R = False

# auction file with list of all tradable point for given period  
CLUSTERS_sourcesink = r"Z:\MISO\04 FTR\02 FTR Auction Models\2020_04\SourcesandSinks_Apr20_AUCTION_Apr20Auc.csv" #########################################

# file with groups nodes no used to one path
# if use_EESL = 1 (use), = 0 (no use)
use_EESL = 0
path_to_fit_EESL = r""

# path to files with constraint-to-node influences (dipoles for example)
CLUSTERS_influence_lib = r"Z:\MISO\06 Libraries\Influences\Dipoles\3_Month_Dipoles"
# additional inf lib for hubs
CLUSTERS_influence_lib_additional = ""#r"Z:\Test\Cluster\data\MISO\2019_09_test\dipoles_MISO"

# file of stage 1 RCTool - used for clusterization
CLUSTERS_RCTool =  r"Z:\tmp\1182\MISO\RCTool_Stage_One.txt" # rez file after run RC_tool for finding same tradable points  #########################################

# calc_F_main = 0 - calc with full list of constraints
# calc_F_main = 1 - calc with only main constraints
# calc_F_main = 3 - calc with min sum for full list of constraints
# calc_F_main = 4 - calc with main constr and additional constraints list for FV
calc_F_main = 0
# use if calc_F_main == 4
FILE_CONSTRAINTS_FOR_FV = r"Z:\Test\Cluster\data\MISO\2019_09\Constr_table\New_mains\MAIN_Constraints_Sep_ALL.csv"

# collateral file if needed

# if true or 1 - make collateral for ech path = 1$
# if false or 0 - use collateral in calc from files
IS_UNITARY_COLLATERAL = True 
CLUSTER_collateral_on = r"Z:\Dropbox (PowerMarketsUR)\Moscow Office\ERCOT\Bids\2019_10\Authopas\New folder\colla3.csv"
CLUSTER_collateral_off = r"Z:\SPP\04 FTR\Auction_Models\2019_10\TCR_REF_PRICES_10-01-2019_11-01-2019_October_Off_Peak.csv"
CLUSTER_collateral_we = r'Z:\SPP\04 FTR\Auction_Models\2019_03\TCR_REF_PRICES_03-01-2019_04-01-2019_March_Off_Peak.csv'
min_collateral_value = 300 # min value of collateral (if path collateral < min_collateral_value, it will be = min_collateral_value)

# Obligation or Option (if IS_OPTION = True - option mode)
IS_OPTION = False

 #  if true - do not use hubs - trad points which have more than one bus in source-sink file
IS_REMOVE_HUBS = False 

# take paths from portfolio
hack_take_portfolio_from_file = 0
start_portfolio_file = r"Z:\Test\Cluster\data\MISO\2019_09_test_6th_stage\portfolio_825_bids.csv"

min_price_delta = 100   # (for low prices): (FV - price) should be > min_price_delta
min_price_fraction_delta = 0.5 # (for high prices): (FV - price) should be > min_price_fraction_delta*price

# ------------------- additional parameters ----------------

DO_FULL_OUTPUT_FILES = True # if true - write portf/constr files on intermediate steps, else - only final !!! (False -> +10% speed run)
IS_WRITE_NEG_CONSTR_COMMENT = True # if true - write to portf file comment of neg constr which blocks path 

DO_F_NEG_SPLIT_OUTPUT = False # true - write files with all info about F for paths with F < 0

hack_take_only_main_constr_paths = False # use only paths wich take main constraints

IS_REMOVE_NODES = False
FILE_NODES_TO_REMOVE = r"C:\Users\Администратор\Desktop\WolframProjects\Data\Nodes_to_use.csv______" # needed column - 'Node'

# file in format 'Node' 'Type'(Source/Sink/All) 'Operation'(Use/Remove)
IS_USE_PATHS_FROM_FILE = False
SOURCE_SINK_TO_USE_FILE = r"C:\Users\Администратор\Desktop\WolframProjects\Data\Nodes_to_use.csv______"

# in loop of main constr choose constr to process by: true - by FV, false - by N of points 
IS_CHOOSE_CONSTR_BY_FV = True 

# true - use specific format for price file and calculate price using prices for periods
# false - use normal price file and price then = source - sink
IS_ANNUAL_PRICES = False 
ANNUAL_PRICE_MODE = 'MEDIAN' # could be 'MAX', 'MEDIAN', 'MIN'

IS_DIVIDE_QUARTAL_PRICES = False # if true - divide all prices by 3

CLUSTER_F_correction_coef = 1.0

# -------------- limits -----------------------------------

# price limits 
# take only paths with price < hack_price_TOP_value
hack_price_TOP_limit = 0 # 0 - off, 1 - on
hack_price_TOP_value = 1000

# take only paths with price > hack_price_BOT_value
hack_price_BOT_limit = 0 # 0 - off, 1 - on
hack_price_BOT_value = -1000 

# collateral limits
# take only paths with collateral < hack_collat_TOP_value
hack_collat_TOP_limit = 0 # 0 - off, 1 - on
hack_collat_TOP_value = 1000

# take only paths with collateral > hack_collat_BOT_value
hack_collat_BOT_limit = 0 # 0 - off, 1 - on
hack_collat_BOT_value = -200 

# take only paths with FV > hack_FV_BOT_value
hack_FV_BOT_limit = 0 # 0 - off, 1 - on
hack_FV_BOT_value = 100

# influence limits
min_influence_for_node = 3 # min inf for take constr for node (for stage1,2 in %)
min_influence_for_path = 5 # min inf for take constr for path (for stage1,2 in %)
max_count_nodes = 2000 # max count of nodes wich has non zero inf on given constraint

mws_max_path = 6 # max MWs for path
mws_max_node = 20 # max MWs for tradable point
MWs_step = 2  # add MWs on each step
n_plus_step = 2  # step for N+

min_corr_value = 0.99999  # min correlation value to assume that nodes are in same cluster

min_neg_influence_to_clear = 0.005#0.005   # if influence less then set value to zero
min_pos_influence_to_clear = 0.005#0.03    # if influence less then set value to zero

min_add_price = 0    # if =1 add to real price max of price_percent*price or min_value_price
price_percent = 0.2    # (for high prices) if min_add_price == 1 add to real price max of price_percent*price
min_value_price = 100    # (for low prices) if min_add_price == 1 add to real price max of min_value_price

iLoopMax = 10000      # max amount of loops
minChange = 1         # count of paths to take in one loop


if class_mode == 1: # peak
    CLUSTER_collateral = CLUSTER_collateral_on
elif class_mode == 0: # off-peak:
    CLUSTER_collateral = CLUSTER_collateral_off
elif class_mode == 2: # peak we:
    CLUSTER_collateral = CLUSTER_collateral_we

CLUSTER_ref_price_file = CLUSTER_collateral

    
if __name__ == "__main__":
    curVer='2020-04-28-1'
    CheckToolVersion('allpaths')
    run_path_optimizer(CLUSTERS_iso, OUTPUT_PATH, class_mode, CLUSTERS_constr_table, CLUSTERS_sourcesink, IS_REMOVE_HUBS, CLUSTERS_RCTool, coef_sigma_down, ignor_3minFTR_for_down_price, min_corr_value, CLUSTERS_influence_lib, CLUSTERS_influence_lib_additional, n_plus_step, CLUSTERS_shadow_price, CLUSTERS_price_distribution_file, min_price_delta, min_price_fraction_delta, CLUSTER_collateral, IS_UNITARY_COLLATERAL, min_collateral_value, min_add_price, price_percent, min_value_price, min_pos_influence_to_clear, min_neg_influence_to_clear, IS_CALC_PRICE_FROM_MEAN_SIGMA, IS_OPTION, FILE_CONSTRAINTS_FOR_FV, IS_REMOVE_NODES, FILE_NODES_TO_REMOVE, IS_ANNUAL_PRICES,  ANNUAL_PRICE_MODE, IS_USE_PATHS_FROM_FILE, SOURCE_SINK_TO_USE_FILE, IS_DIVIDE_QUARTAL_PRICES, IS_OFF_STAGE_R, CLUSTER_F_correction_coef, False, hack_take_only_main_constr_paths, max_count_nodes, min_influence_for_node, min_influence_for_path, path_to_fit_EESL, use_EESL, hack_FV_BOT_limit, hack_FV_BOT_value, hack_price_TOP_limit, hack_price_TOP_value, hack_price_BOT_limit, hack_price_BOT_value, hack_collat_TOP_limit, hack_collat_TOP_value, hack_collat_BOT_limit, hack_collat_BOT_value, DO_F_NEG_SPLIT_OUTPUT, MWs_step, hack_take_portfolio_from_file, start_portfolio_file, iLoopMax, IS_CHOOSE_CONSTR_BY_FV, IS_WRITE_NEG_CONSTR_COMMENT, mws_max_path, calc_F_main, mws_max_node, minChange, coef_recent_neg, IS_CHECK_MinDART_RECENT, DO_FULL_OUTPUT_FILES, IS_DOWN_PRICE, USE_MIN_RT_DA, min_RT, min_DA)