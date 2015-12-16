############################### Import packages ##############################
import os
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

############################### Define Globals ##############################
ROOT = os.path.dirname(os.path.realpath('__file__'))
DATA = ROOT + '/Data/'
SUBM = ROOT + '/Submissions/basic_XGB/'
############################### Define functions ##############################

def feat_importances(frst, feats):
    outputs = pd.DataFrame({'feats': feats,
                            'weight': frst.feature_importances_})
    outputs = outputs.sort(columns='weight', ascending=False)
    print outputs


def create_val_and_train(df, seed=42,  split_rt = .20):
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

def add_prediction(df, model):
    df_x = df.drop('QuoteConversion_Flag', axis=1).values
    return model.predict_proba(df_x)[:, 1]

def load_append(PATH):
    train = pd.read_csv(PATH + 'clean_train.csv')
    test = pd.read_csv(PATH + 'clean_test.csv')
    return train, test

############################### Executions ##############################

train, test = load_append(DATA)
all_preds = []
all_scores = 0
for i, seed in enumerate([42, 417]):
    modeling, val = create_val_and_train(train, seed=seed)
    X = modeling.drop('QuoteConversion_Flag', axis=1).values
    y = modeling.QuoteConversion_Flag.values

    clf = xgb.XGBClassifier(n_estimators=550, nthread=-1, max_depth=6,
                            learning_rate=0.025, silent=True, subsample=0.7,
                            colsample_bytree=0.8)

    xgb_model = clf.fit(X, y, eval_metric="auc")

    name = 'pred' + str(seed)

    all_preds.append(name)
    val[name] = add_prediction(val, xgb_model)
    test[name] = add_prediction(test, xgb_model)

    score = roc_auc_score(val.QuoteConversion_Flag, val[name])
    all_scores += score
    print "Score for seed {0}: {1}".format(seed, score)
    print "Overall after {0} loops: {1}".format(i+1, all_scores/(1+i))

submission_title = 'running_deep_score{0}.csv'.format(all_scores)
keep_cols = ['QuoteNumber', 'QuoteConversion_Flag']
test['QuoteConversion_Flag'] = test[all_preds].mean(axis=1)
test[keep_cols].to_csv(SUBM + 'finding_baseline.csv', index=False)




