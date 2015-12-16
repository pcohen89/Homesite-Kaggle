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
############################### Define functions ##############################

train = pd.read_csv(DATA + 'train.csv')
test = pd.read_csv(DATA + 'test.csv')

train['is_train'] = 1
test['is_train'] = 0
all_df = train.append(test, ignore_index=True)

# Lets play with some dates
all_df['Date'] = pd.to_datetime(pd.Series(all_df['Original_Quote_Date']))
all_df = all_df.drop('Original_Quote_Date', axis=1)

all_df['Year'] = all_df['Date'].apply(lambda x: int(str(x)[:4]))
all_df['Month'] = all_df['Date'].apply(lambda x: int(str(x)[5:7]))
all_df['weekday'] = all_df['Date'].dt.dayofweek

all_df = all_df.drop('Date', axis=1)
all_df = all_df.fillna(-1)

for col in all_df.columns:
    if all_df[col].dtype == 'object':
        print(col)
        lbl = preprocessing.LabelEncoder()
        all_df[col] = lbl.fit_transform(list(all_df[col].values))

all_df[all_df.is_train == 1].to_csv(DATA + 'clean_train.csv', index=False)
all_df[all_df.is_train == 0].to_csv(DATA + 'clean_test.csv', index=False)


