import pandas as pd
import numpy as np


def handle_invalid_negatives(df):
    """
    Replace zero and negative values with NaN for columns where 
    these values are physically meaningless.
    """

    invalid_negative_features = ['KSOL','HLM CLint','MLM CLint',
                                 'MPPB','MBPB','MGMB']

    for feature in invalid_negative_features:
        df[feature] = df[feature].apply(lambda x: x if x > 0 else np.nan)
    return df


def log_transform():
    
    return


def impute():
    
    return


def preprocess():
    
    return

if __name__ == '__main__':
    df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv")
    df_clean = preprocess(df)
    print(df_clean.head())
    print(df_clean.shape)
    print(df_clean.isnull().sum())