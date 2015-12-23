############################### Import packages ##############################
import os
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier

############################### Define Globals ##############################
ROOT = os.path.dirname(os.path.realpath('__file__'))
DATA = ROOT + '/Data/'
SUBM = ROOT + '/Submissions/pile_o_logistics/'
############################### Define functions ##############################

def feat_importances(frst, feats):
    outputs = pd.DataFrame({'feats': feats,
                            'weight': 100*frst.feature_importances_})
    outputs = outputs.sort(columns='weight', ascending=False)
    return outputs


def create_val_and_train(df, seed=42,  split_rt=.20):
    """
        Creates two samples (generally used to create
        train and validation samples)
        Parameters
        ----------------------------------------------------
        ids: this is the level of randomization, so if you want to
        randomize countries, rather than cities, you would
        set this to 'countries'
        split_rate: pct of data to assign as validation
        Output
        ----------------------------------------------------
        trn_for_mods (1-split_rate of df), trn_for_val (split_rate of data)
    """
    np.random.seed(seed)
    # Create random vector to split train val on
    vect_len = len(df.ix[:, 0])
    df['rand_vals'] = (np.array(np.random.rand(vect_len,1)))
    # split data into two dfs
    trn_for_mods = df[df.rand_vals > split_rt]
    trn_for_val = df[df.rand_vals <= split_rt]
    # drop rand_vals
    trn_for_val = trn_for_val.drop('rand_vals', axis=1)
    trn_for_mods = trn_for_mods.drop('rand_vals', axis=1)
    return trn_for_mods, trn_for_val

def create_feat_list(df, non_features):
    feats = list(df.columns.values)
    for var in non_features:
        feats.remove(var)
    return feats


def load_append(PATH):
    train = pd.read_csv(PATH + 'clean_train.csv')
    test = pd.read_csv(PATH + 'clean_test.csv')
    return train, test


def normalize_cols(df):
    for col in df.columns:
        if col in ['QuoteConversion_Flag', 'QuoteNumber']:
            continue
        elif df[col].std() > 0:
            df[col] = (df[col]-df[col].mean())/df[col].std()
        else:
            df[col] = 0
            print col
    return df

def select_cols(col_list, pct=.4):
    chosen = [(x < pct)[0] for x in list(np.random.rand(len(col_list), 1))]
    return list(col_list[chosen])

############################### Executions ##############################
train, test = load_append(DATA)
train, test = normalize_cols(train), normalize_cols(test)

for col in train.columns:
    stats = train[col].describe()
    print train[[col, 'QuoteConversion_Flag']].corr().ix[0, 1]
    if train[[col, 'QuoteConversion_Flag']].corr().ix[0, 1] < 0:
        train[col] *= -1
        test[col] *= -1

feats = select_cols(train.columns, 1)
feats.remove('QuoteConversion_Flag')
frst = RandomForestClassifier(n_estimators=400, max_depth=30, n_jobs=-1)
frst.fit(train[feats], train.QuoteConversion_Flag)
weights = feat_importances(frst, feats)

for i, feat in enumerate(list(weights.feats[0:15])):
    for feat2 in list(weights.feats[(i+1):15]):
        train[feat + 'x' + feat2] = train[feat] * train[feat2]
        test[feat + 'x' + feat2] = test[feat] * test[feat2]
        feats.append(feat + 'x' + feat2)


frst = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1)
frst.fit(train[feats], train.QuoteConversion_Flag)
weights = feat_importances(frst, feats)

train.to_csv(DATA + 'train_wints.csv', index=False)
test.to_csv(DATA + 'test_wints.csv', index=False)

