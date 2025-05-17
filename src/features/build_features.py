import pandas as pd
from datetime import datetime
import category_encoders as ce
import pickle

# create dummy features or encoding
def create_dummy_vars(df):
    """Perform feature engineering and encoding"""
    
    # Calculate 'Car Age' feature based on Prod. Year
    df['Car Age'] = datetime.now().year - df['Prod. year']
    df.drop(columns=['Prod. year'], inplace=True)
    
    # Drop features with low importance or that can be implied by Make&Model
    df = df.drop('Doors', axis=1)
    df = df.drop('Airbags', axis=1)
    df = df.drop('Wheel', axis=1)
    df = df.drop('Leather interior', axis=1)
    df = df.drop('Category', axis=1)
    df = df.drop('Color', axis=1)

    # Before encoding, save cleaned file with no target variable for Streamlit to consume
    cleaned_df = df.copy().drop('Price', axis=1)
    cleaned_df.to_csv('data/processed/Cleaned_Car_Price.csv', index=None)
    
    # Create dummy variables for low cardinality 'object' type variables
    df = pd.get_dummies(df, 
                        columns=[
                                    'Manufacturer', 
                                    'Fuel type', 
                                    'Engine volume',
                                    'Gear box type', 
                                    'Drive wheels'
                                    ], 
                        dtype=int)

    # Separate the input features and target variable
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Apply LOO encoding to high-cardinality categorical column 'Model'
    loo_encoder = ce.LeaveOneOutEncoder(cols=['Model'])
    X['Model'] = loo_encoder.fit_transform(X[['Model']], y)
    # Save trained encoder model for Streamlit to consume
    with open('models/loo_encoder.pkl', 'wb') as f:
        pickle.dump(loo_encoder, f)

    # Save processed data
    processed_df = X.copy()
    processed_df['Price'] = y
    processed_df.to_csv('data/processed/Processed_Car_Price.csv', index=None)

    return X, y