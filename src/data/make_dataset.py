
import pandas as pd

def load_and_preprocess_data(data_path):
    """Clean dataset"""
    '''
        1. Extract numbers from age column
        2. Impute missing age based on median age for each graduation year
        3. Remove outliers in age
        4. Fill missing gender with 'unknown'
    '''
            
    # Import the data
    df = pd.read_csv(data_path)
   
    # Drop record if graduation year is missing
    df = df.dropna(subset=['gradyear'])
    
    # Extract age as integers (discard decimal places and random invalid values ex. '19.Mai')
    df['age'] = df['age'].astype(str).str.extract(r'(\d+)', expand=False)
     
    # Impute missing 'age' based on median age for each gradyear
    df_nonnull = df[df['age'].notnull()].copy()
    df_nonnull['age'] = df_nonnull['age'].astype(int)
    median_age_per_year = df_nonnull.groupby('gradyear')['age'].median()
    
    df['age'] = df['age'].fillna(df['gradyear'].map(median_age_per_year))
    df['age']= df['age'].astype(int)
    
    # Remove outliers in age - only keep data in 1 - 99 percentile range
    lower = df['age'].quantile(0.01)
    upper = df['age'].quantile(0.99)
    df = df[(df['age'] >= lower) & (df['age'] <= upper)]
    
    # Impute missing 'gender'
    df['gender'] = df['gender'].fillna('unknown')

    return df
