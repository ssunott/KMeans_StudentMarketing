from sklearn.preprocessing import StandardScaler
import pickle

def gender_to_numeric(x):
    if x=="M":
        return 1
    if x=="F":
        return 2
    if x== 'unknown':
        return 3

# create dummy features or encoding
def create_dummy_vars(df):
    """Perform feature engineering and encoding"""

    # Apply encoding to 'gender' and scale the keyword features
    scaled_df = df.copy()
    scaled_df['gender'] = scaled_df['gender'].apply(gender_to_numeric)

    keyword = scaled_df.columns[4:40]
    features = scaled_df[keyword]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_df[keyword] = features

    scaled_df.to_csv('data/processed/Processed_Cluster_Marketing.csv', index=None)
   
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
   
    return scaled_df  