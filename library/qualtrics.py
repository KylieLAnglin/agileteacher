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

