import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors



def handle_censored_values(df):
    """
    Some values in the test dataset contain non-precise values, denoted with '<' or '>'. 
    To handle these, the symbols will be dropped, and a new boolean column will be created to denote te original symbol, such to not lose any 
    information or make assumptions
    """

    ignore_features = ['SMILES','Molecule Name', 'INCHIKEY']

    for feature in df.columns:
        if feature in ignore_features or 'modifier' in feature.lower():
            continue
        else:
            df[f'{feature}_censored'] = df[feature].apply(
                lambda x: 0 if isinstance(x,str) and '>'
                else  1 if isinstance(x,str) and '<'
                 else False
                ) #First checks if '>' or '<' is in column, and creates a boolean columns if True
            df[feature] = df[feature].apply(lambda x: float(x.replace('>', '').replace('<', '').strip()) if isinstance(x,str) else x) #Drops '<' or '>' 

    return df

def handle_invalid_negatives(df):
    """
    Replace zero and negative values with NaN for columns where 
    these values are physically meaningless.   
    """

    invalid_negative_features = ['KSOL','HLM CLint','MLM CLint',
                                 'MPPB','MBPB','MGMB']

    for feature in invalid_negative_features:
        if feature in df.columns:
            df[feature] = df[feature].apply(lambda x: x if x > 0 else np.nan)
    return df


def log_transform(df):
    """
    Log transform the features which are highly skewed to achieve normal distributions 
    """

    skewed_features = ['HLM CLint', 'MLM CLint', 'Caco-2 Permeability Efflux']

    for feature in skewed_features:
        if feature in df.columns:
            df[f'log_{feature}'] = df[feature].apply(lambda x: np.log(x)) #rename coloumns to preserve original columns
            df = df.drop(columns=[feature]) #drop original feature to prevent data leakage
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


def return_collinear_features(df):
    """
    Return a list of collinear features if their correlation exceeds threshold value.
    The user can then decide which feature to drop.
    """
    float_columns = df.select_dtypes(float) #use only the columns of floats
    correlation_matrix = float_columns.corr() #compute correlation matrix of all features
    upper_corr_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)) #define the upper triangle of the correlation matrix to avoid redunancy
    threshold = 0.85 #threshold chosen as a conservative value due to small dataset size
    collinear_pairs = [] #define array to store pairs that are determined to be correlated

    # --determine correlation loop
    for col in upper_corr_matrix.columns: 
        correlated = upper_corr_matrix[col][upper_corr_matrix[col] > threshold].index.tolist() #for every column in the upper trianlge, it is correlated if the correlation exceeds threshold
        for feature in correlated: #loop over each correlated feature pairs to store them 
            collinear_pairs.append((col,feature))

    return collinear_pairs


def assign_molecular_features(df):
    """
    Calculate molecular properties from the SMILES string and assign them to new columns.
    Properties: -Mol. Weight (g/mol)
                -Hbonding Donors
                -Hbonding Acceptors
                -LogP
                -Rotatable Bonds
                Topological Polar Surface Area (TPSA)
    """
    mols = df['SMILES'].apply(Chem.MolFromSmiles) 

    #-- Descriptors added as named columns
    df['MolWt'] = mols.apply(Descriptors.MolWt)
    df['TPSA'] = mols.apply(Descriptors.TPSA)
    df['HBD'] = mols.apply(Descriptors.NumHDonors)
    df['HBA'] = mols.apply(Descriptors.NumHAcceptors)
    df['RotBonds'] = mols.apply(Descriptors.NumRotatableBonds)
    df['LogP'] = mols.apply(Descriptors.MolLogP)

    #-- Generate and assign Morgan Fingerprints
    num_bits = 1024
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=num_bits) #create Fingerprint generator
    fps = mols.apply(lambda mol: np.array(generator.GetFingerprintAsNumPy(mol)))     #Apply generator to molecules
    df_fps = pd.DataFrame(fps.tolist(), columns=[f'fps_{i}' for i in range(num_bits)], index=df.index)
    df = pd.concat([df,df_fps],axis=1)
    return df


def preprocess(df):
    df = handle_censored_values(df)
    df = handle_invalid_negatives(df)
    df = log_transform(df)
    df = impute(df)
    df = assign_molecular_features(df)

    return df

if __name__ == '__main__':
    df = pd.read_csv("data/raw/train_raw.csv") 
    df_clean = preprocess(df)
    print(df_clean.head())
    print(df_clean.shape)
    print(df_clean.isnull().sum())