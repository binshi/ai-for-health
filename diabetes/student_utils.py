import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df = df.join(ndc_df.set_index('NDC_Code'), on='ndc_code')
    df.drop(['Non-proprietary Name', 'Dosage Form', 'Route Name', 'Company Name', 'ndc_code', 'Product Type'], axis = 1, inplace = True) 
    df.rename(columns={'Proprietary Name': 'generic_drug_name'}, inplace = True)
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    fc=df.groupby('encounter_id').nth(0).reset_index()
    sorted=fc.sort_values('encounter_id')
    first_encounter_df = sorted.groupby('patient_nbr').nth(0).reset_index()
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size = round(total_values * (1 - 0.2))
    train_size = round(sample_size * (1 - 0.25))
    train = df[df[patient_key].isin((unique_values[:sample_size])[:train_size])].reset_index(drop=True)
    validate = df[df[patient_key].isin((unique_values[:sample_size])[train_size:])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[sample_size:])].reset_index(drop=True)
    return train, validate, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        vocab = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        one_hot_feature = tf.feature_column.indicator_column(vocab)
        output_tf_list.append(one_hot_feature)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean = MEAN, std = STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key = col, default_value = 0, normalizer_fn = normalizer, dtype = tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    df['label'] = df['pred_mean'].apply(lambda x: 1 if x>=5 else 0 )
    student_binary_prediction=df['label'].to_numpy()
    return student_binary_prediction
