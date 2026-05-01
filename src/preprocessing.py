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


def log_transform(df):
    """
    Log transform the features which are highly skewed to achieve normal distributions 
    """

    skewed_features = ['HLM CLint', 'MLM CLint', 'Caco-2 Permeability Efflux']

    for feature in skewed_features:
        df[f'log_{feature}'] = df[feature].apply(lambda x: np.log(x)) #rename coloumns to preserve original columns

    return df


def impute(df):
    """
    Impute missing values if less than 50% are missing. If a column is missing more than 50% of values, it is dropped 
    """

    impute_cutoff = 0.50 #cutoff for acceptable percentage of missing values (50%)

    for feature in df:
        if df[feature].dtype == 'string' or df[feature].dtype == 'object': #these features aren't logical to impute (structural descriptors or modifiers)
            continue

        percent_missing = (df[feature].isna().sum()) / len(df[feature]) #calculate percentage of column values missing to compare with skip_impute
        if percent_missing > impute_cutoff:
            df.drop(columns = [feature], inplace=True) #drop the columns which have more than half of their values missing

        #--- Determine imputation method based on skewness of data (if data is reasonably normally distributed -> mean, if its skewed -> median)
        elif abs(df[feature].skew()) < 0.5:
            df[feature] = df[feature].fillna(df[feature].mean())

        elif abs(df[feature].skew()) > 0.5:
            df[feature] =  df[feature].fillna(df[feature].median())

    return df


def preprocess():
    
    return

if __name__ == '__main__':
    df = pd.read_csv("hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv")
    df_clean = preprocess(df)
    print(df_clean.head())
    print(df_clean.shape)
    print(df_clean.isnull().sum())