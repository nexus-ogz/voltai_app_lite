import joblib
import seaborn as sns
import json, pickle, sys, os
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, streamlit as st


path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(path, '../..'))

from utils.fonctions import load_parquet_data, load_pickle
import zipfile

def exception_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    return wrapper

def load_pickle_zipped(file_path, type='joblib'):
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            for file_name in z.namelist():
                with z.open(file_name) as f:
                    if type=='joblib':
                        return joblib.load(f)
                    elif type=='pickle':
                        return pickle.load(f)
                    else:
                        raise Exception("Underneath file not handled")
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except zipfile.BadZipFile:
        st.error(f"Invalid zip file: {file_path}")
        return None

load_pickled_data_cache = st.cache_data(load_pickle)
load_pickled_model_cache = st.cache_resource(load_pickle)
load_pickled_zipped_model_cache = st.cache_resource(load_pickle_zipped)

def fonction_communes_pages():
    pass

@st.cache_resource
def load_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None