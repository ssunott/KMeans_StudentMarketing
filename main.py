from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_KMeanmodel

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/Clustering_Marketing.csv"
    df = load_and_preprocess_data(data_path)
           
    # Create dummy variables and separate features and target
    df = create_dummy_vars(df)

    # Train the k-means model
    train_KMeanmodel(df)
