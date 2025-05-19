import pandas as pd


# create dummy features or encoding
def create_dummy_vars(df):
    """Perform feature engineering and encoding"""
    # Extract age as integers (discard decimal places and random invalid values ex. '19.Mai')
    df['age'] = df['age'].astype(str).str.extract(r'(\d+)', expand=False)
     
    # Impute missing 'age' based on median age for each gradyear
    df_nonnull = df[df['age'].notnull()].copy()
    df_nonnull['age'] = df_nonnull['age'].astype(int)
    median_age_per_year = df_nonnull.groupby('gradyear')['age'].median()
    
    df['age'] = df['age'].fillna(df['gradyear'].map(median_age_per_year))
    df['age']= df['age'].astype(int)
    
    # Remove outliers in age - only keep data in 1st to 99th percentile range
    lower = df['age'].quantile(0.01)
    upper = df['age'].quantile(0.99)
    df = df[(df['age'] >= lower) & (df['age'] <= upper)]
    
    # Impute missing 'gender'
    df['gender'] = df['gender'].fillna('unknown')
    
    # Create dummy variables for low cardinality 'object' type variables
    df = pd.get_dummies(df, columns=['gender'], dtype=int)

    df.to_csv('data/processed/Processed_Cluster_Marketing.csv', index=None)

    return df