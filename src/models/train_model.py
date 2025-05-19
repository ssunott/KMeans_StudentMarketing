from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from src.visualization.visualize import plot_silhouette_score, plot_elbow_score

# Function to train the model and return test data
def train_KMeanmodel(X):
    """Train the KMeans model and return test data"""

    # Find best k using silhouette score
    best_k = None
    best_score = -1
    scores = {}

    for k in range(2, 11): # to loop
        ypred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score = silhouette_score(X, ypred)
        scores[k] = score
        print(f"k={k:2d}  silhouette score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k by silhouette score: {best_k} (score={best_score:.4f})")
    # Plot and save silhouette scores image
    plot_silhouette_score(scores)
    
    # Find best k using elbow method
    inertias = []
    K = range(1, 10)   # try k from 1 to 10
    for k in K:
        wcss = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(wcss.inertia_)

    print(f"\nBest k by elbow method: {K[inertias.index(min(inertias))]} (inertia={min(inertias):.4f})")
    # Plot and save elbow scores image
    plot_elbow_score(K, inertias)

    # 4. Train final model with best_k
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    model_labels = model.fit_predict(X)
    X['cluster'] = model_labels

    # 5. Save the scaler and model for later use
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(model, f)
  
    X.to_csv("data/processed/clustered_data.csv", index=False)
