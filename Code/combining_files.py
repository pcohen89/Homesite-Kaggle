############################### Import packages ##############################
import os
import pandas as pd

############################### Define Globals ##############################
ROOT = os.path.dirname(os.path.realpath('__file__'))
SUBM_PATH = ROOT + '/Submissions/'
############################### Define functions ##############################


def subm_correl(subm_path, subm_path2, id, target):
    """
    Measures correlation between to Kaggle submissions
    """
    subm1 = pd.read_csv(SUBM_PATH + subm_path)
    subm2 = pd.read_csv(SUBM_PATH + subm_path2)
    subm2 = subm2.rename(columns={target: 'target2'})
    merged_df = subm1.merge(subm2, on=id)
    return merged_df.corr()


def merge_subms(subm_dict, path, name, target, id):
    """
    :param subm_dict: Dictionary of dfs to merge, where key is csv name and
    value is weight (values must sum to 1 to keep outcome in original range
    :param path: path to submission folder
    :param name: name of new file
    :param target: outcome variable of submission
    :return:
    """
    subm = pd.read_csv(path+'template.csv')
    for csv, weight in subm_dict.iteritems():
        # Read in a new csv
        score = pd.read_csv(path+csv)
        # rename target to avoid merge issues
        score = score.rename(columns={target: 'target2'})
        # Merge files to be averaged
        subm = subm.merge(score, on=id)
        subm[target] += weight * subm['target2']
        subm = subm.drop('target2', 1)
    subm.to_csv(path+name, index=False)

def check_weight_and_merge(dict, name):
    """
    :param dict: file, weight pairs
    :param name: name of resulting blended file
    :return: blended file saved to server
    """
    total_weight = 0
    for key, val in dict.iteritems():
        total_weight += val
    print "The total weight should be 1.0, it is: %s" % (total_weight)
    merge_subms(dict, SUBM_PATH, name, 'cost')

############################### Executions ##############################
subm_correl('/basic_XGB/running_3deep_score9665.csv',
            '/pile_o_logistics/running_10logitsticbetter_score9441.csv',
            'QuoteNumber',
            'QuoteConversion_Flag')

