import os
import pandas as pd


def extract_column_labels(csv_path: str):
    """imports qualtrics csv and returns dict /
    containing column labels

    Args:
        csv_path (str): path to csv including file name 

    Returns:
        [dict]: dictionary mapping column names to survey q's
    """
    survey = pd.read_csv(csv_path, nrows=2)

    survey_cols = survey[0:1].to_dict('series')

    survey_labels = {}
    for key in survey_cols:
        survey_labels[key] = survey_cols[key][0]

    return survey_labels


def select_valid_rows(csv_path:str):
    """import qualtrics csv and select rows with survey responses

    Args:
        csv_path (str): path to csv including file name 

    Returns:
        [pd.DataFrame]: dataframe containing only survey responses
    """
    survey = pd.read_csv(csv_path)
    survey = survey[2:]

    return survey

def drop_meta_data(df:pd.DataFrame):
    """Select columns containing survey responses

    Args:
        df (pd.DataFrame): df containing qualtrics data with original column names

    Returns:
        [pd.DataFrame]: df only containing survey response columns, no meta-data

    Survey response column names begin with Q 
    """
    filter_col = [col for col in df if col.startswith('Q')]
    df = df[filter_col]

    return df


def search_column_labels(column_labels:dict, search_term: str, print: bool=False):
    """searches label dictionary for word(s)

    Args:
        column_labels (dict): dictionary with keys containing col names and values containing survey questions
        search_term (str): word to search values for

    Function is designed to take dict resulting from extract_column_labels() as argument

    Returns:
        list: List of column names containing search term
    """
    cols = []
    for key, value in column_labels.items():
        if search_term in value:
            if print:
                print(key, value)
            cols.append(key)
    
    return cols
