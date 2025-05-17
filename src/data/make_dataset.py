
import pandas as pd

def load_and_preprocess_data(data_path):
    """Clean dataset"""
    
    # Import the data from 'realtor-data.csv'
    df = pd.read_csv(data_path)

     # Drop rows with missing target variable
    df = df.dropna(subset=['Price'])
    
    # Drop unnecessary features
    df = df.drop('ID', axis=1)
    df = df.drop('Levy', axis=1)
    
    # Convert 'Mileage' to numerical value
    df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False).astype(float)
        
    # Remove outliers in price - only keep data in 1st to 99th percentile range
    lower = df['Price'].quantile(0.01)
    upper = df['Price'].quantile(0.99)
    df = df[(df['Price'] >= lower) & (df['Price'] <= upper)]
    
    # Only keep cars produced after 1990
    df = df[(df['Prod. year'] >= 1990)]
        
    return df
