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
    print outputs


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

def add_preds(v, t, model):
    val_preds = model.predict_proba(v)[:, 1]
    test_preds = model.predict_proba(t)[:, 1]
    return val_preds, test_preds


def add_preds_noproba(val, test, model):
    val_preds = model.predict(val)
    test_preds =  model.predict(test)
    return val_preds, test_preds


def load_append(PATH):
    train = pd.read_csv(PATH + 'clean_train.csv')
    test = pd.read_csv(PATH + 'clean_test.csv')
    return train, test


def handle_scores(all_scores, preds):
    score = roc_auc_score(val.QuoteConversion_Flag, val[preds])
    all_scores += score
    print "Score for seed {0}: {1}".format(seed, score)
    return all_scores


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


def create_x_and_y(df):
    drop_cols = ['QuoteConversion_Flag', 'is_train', 'QuoteNumber']
    X = df.drop(drop_cols, axis=1)
    y = df.QuoteConversion_Flag
    return X, y


############################### Executions ##############################

train, test = load_append(DATA)
# Normalize columns
train = normalize_cols(train)
test = normalize_cols(test)

all_preds = []
all_scores = 0
full_list = [42, 417, 1, 2, 3, 4, 5]
for i, loop in enumerate([42]):
    mod_train, val = create_val_and_train(train, seed=loop, split_rt=.15)
    frst_X, frst_y = create_x_and_y(mod_train)

    feats = select_cols(frst_X.columns, .95)

    val_X = val[feats]
    test_X = test[feats]

    frst = RandomForestClassifier(n_estimators=400, max_depth=20, n_jobs=3)
    frst.fit(frst_X[feats], frst_y)
    feat_importances(frst, feats)

    frst1_nm = "frst_preds"
    val[frst1_nm], test[frst1_nm] = add_preds(val[feats], test[feats], frst)

    all_scores = handle_scores(all_scores, frst1_nm)

    all_preds.append(frst1_nm)

    print "Overall after {0} loops: {1}".format(i+1, all_scores/(i+1))


title_score = int(10000*all_scores/(i+1))
submission_title = 'running_10logitwtree_score{0}.csv'.format(title_score)
keep_cols = ['QuoteNumber', 'QuoteConversion_Flag']
test['QuoteConversion_Flag'] = test[all_preds].mean(axis=1)
test[keep_cols].to_csv(SUBM + submission_title, index=False)


