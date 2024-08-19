# Standard library imports
import os
import warnings

# Third party imports
import numpy as np
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support, 
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from prefect import flow, task

# Local application imports
from config import *
from utils import * 
from feature_extraction import FeatureExtraction
from text_processing import TextProcessing

warnings.filterwarnings("ignore")

@task(retries=3, retry_delay_seconds=2,
      name="Text processing task", 
      tags=["pos_tag"])
def text_processing_task(language: str, file_name: str, version: int):
    """This task is used to run the text processing process
    Args:
        language (str): language of the text
        file_name (str): file name of the data
        version (int): version of the data
    Returns:
        None"""
    text_processing_processor = TextProcessing(language=language)
    text_processing_processor.run(file_name=file_name, version=version)

@task(retries=3, retry_delay_seconds=2,
      name="Feature extraction task", 
      tags=["feature_extraction", "topic_modeling"])
def feature_extraction_task(data_path_processed: str, 
                            data_version: int):
    """This task is used to run the feature extraction process
    Args:
        data_path_processed (str): path where the data is stored
        data_version (int): version of the data
    Returns:
        None"""
    feature_extraction_processor = FeatureExtraction()
    feature_extraction_processor.run(data_path_processed=data_path_processed, 
                                     data_version = VERSION)