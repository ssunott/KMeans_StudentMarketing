from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pickle
from src.visualization.visualize import plot_silhouette_score, plot_elbow_score, plot_db
from kneed import KneeLocator


# Function to train the model and return test data
def train_KMeanmodel(X, df):
    """Train the KMeans model and return test data"""

    # 1. Find best k using silhouette score
    best_k = None
    best_score = -1
    scores = {}
    #print(f"\nFinding best k by silhouette score:")
    for k in range(2, 10): 
        ypred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score = silhouette_score(X, ypred)
        scores[k] = score
        #print(f"k={k:2d}  silhouette score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k by silhouette score: {best_k} (score={best_score:.4f})")
    # Plot and save silhouette scores image
    plot_silhouette_score(scores)

    # 2. Find best k using davies_bouldin score
    db_scores = []
    Ks = range(2, 11)
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(X)
        labels = km.labels_
        db = davies_bouldin_score(X, labels)
        db_scores.append(db)

    # find best k (minimum DB index)
    best_k_db = K[db_scores.index(min(db_scores))]
    print("\nBest k by Davies–Bouldin:", best_k_db)
    plot_db(Ks, db_scores, best_k_db)
    
    # 3. Find best k using elbow method
    inertias = []
    K = range(1, 20)   # try k from 1 to 20
    for k in K:
        wcss = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(X)
        inertias.append(wcss.inertia_)
    kl = KneeLocator(K, inertias, curve='convex', direction='decreasing')
    best_k = kl.elbow
    print(f"\nBest k by elbow method: {best_k}")
    # Plot and save elbow scores image
    plot_elbow_score(K, inertias)

    # 4. Train final model with best_k found by elbow score
    model = KMeans(n_clusters=best_k)
    model_labels = model.fit_predict(X)
    X['cluster'] = model_labels
    df['cluster'] = model_labels

    # 5. Save the model for later use
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(model, f)
  
    X.to_csv("data/processed/scaled_clustered_data.csv", index=False)
    df.to_csv("data/processed/clustered_data.csv", index=False)

    return df

