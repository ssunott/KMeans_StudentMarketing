
import pandas as pd

def load_and_preprocess_data(data_path):
    """Clean dataset"""
    
    # Import the data
    df = pd.read_csv(data_path)
   
    # Drop record if graduation year is missing
    df = df.dropna(subset=['gradyear'])

    return df
