
import matplotlib.pyplot as plt


def plot_silhouette_score(scores):
    """
    Plot Silhouette Score of the model.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    score_key = list(scores.keys())
    score_val = list(scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(score_key, score_val, marker='o', linestyle='-')
    plt.xticks(score_key)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")

    plt.legend()
    plt.tight_layout()
    plt.savefig("sscore.png")
    
    
def plot_elbow_score(K, inertias):
    """
    Plot Elbow Score (WCSS) showing of the model.
    
    Args:
        k = range
        inertias = inertia values
    """
    
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertias)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (within‐cluster SSE)')
    plt.title('Elbow Plot')
    plt.savefig("wcss.png")
     

