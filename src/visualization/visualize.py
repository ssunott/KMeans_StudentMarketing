
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
    plt.ylabel('Inertia')
    plt.title('Elbow Plot')
    plt.savefig("wcss.png")
     

def plot_cluster(df):
    """
    Analyze data clusters.
    
    Args:
        df = dataframe after applying trained model
    """
    # get the size of each cluster and total male, female, and unknown gender in entire df
    size = df.groupby('cluster').size()  
    total_female = (df['gender'] == 'F').sum()
    total_male   = (df['gender'] == 'M').sum()
    total_other  = (df['gender'] == 'unknown').sum()

    avg_age = df.groupby('cluster')['age'].mean()
    female = df[df['gender'] == 'F'].groupby('cluster').size() / size * 100
    overall_female = df[df['gender'] == 'F'].groupby('cluster').size() / total_female * 100
    male = df[df['gender'] == 'M'].groupby('cluster').size() / size * 100
    overall_male = df[df['gender'] == 'M'].groupby('cluster').size() / total_male * 100
    other = df[df['gender'] == 'unknown'].groupby('cluster').size() / size * 100
    overall_other = df[df['gender'] == 'unknown'].groupby('cluster').size() / total_other * 100
    avg_friends = df.groupby('cluster')['NumberOffriends'].mean()
    
    cluster_analysis = pd.DataFrame({
        'Average Age': avg_age,
        'Percentage Female': female,
        'Overall Percentage Female': overall_female,
        'Percentage Male': male,
        'Overall Percentage Male': overall_male,
        'Percentage Other': other,
        'Overall Percentage Other': overall_other,
        'Average Number of Friends': avg_friends,
        'Number of Students': size
    })
    
    # print cluster analysis data
    print("Cluster Analysis:")
    for cluster, row in cluster_analysis.iterrows():
        print(f"Cluster {cluster}:")
        print(f"  Average Age: {row['Average Age']:.2f}")
        print(f"  Female% within cluster: {row['Percentage Female']:.2f}%")
        print(f"  Female% across sample: {row['Overall Percentage Female']:.2f}%")
        print(f"  Male% within clsuter: {row['Percentage Male']:.2f}%")
        print(f"  Male% across sample: {row['Overall Percentage Male']:.2f}%")
        print(f"  % of unknown gender in cluster: {row['Percentage Other']:.2f}%")
        print(f"  % of unknown gender across sample: {row['Overall Percentage Other']:.2f}%")
        print(f"  Average Number of Friends: {row['Average Number of Friends']:.2f}")
        print(f"  Number of Students: {row['Number of Students']}")

    # generate plots and save image
    clusters = cluster_analysis.index.to_list()
    n_clusters = len(clusters)
    indices = np.arange(n_clusters)

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    axs[1, 0].bar(clusters, cluster_analysis['Average Age'], color='skyblue')
    axs[1, 0].set_title('Average Age by Cluster')
    axs[1, 0].set_ylim(14, 18)

    axs[2, 0].bar(clusters, cluster_analysis['Average Number of Friends'], color='salmon')
    axs[2, 0].set_title('Average Number of Friends by Cluster')

    axs[0, 0].bar(clusters, cluster_analysis['Number of Students'], color='gold')
    axs[0, 0].set_title('Number of Students by Cluster')
    
    # Overall Gender Percentage per Cluster
    genders = ['Female', 'Male', 'Unknown']
    data = {}
    for cluster in cluster_analysis.index:
        data[f'Cluster {cluster}'] = [
            cluster_analysis.loc[cluster, 'Overall Percentage Female'],
            cluster_analysis.loc[cluster, 'Overall Percentage Male'],
            cluster_analysis.loc[cluster, 'Overall Percentage Other']
        ]
    df_plot = pd.DataFrame(data, index=genders)

    df_plot.plot(
        kind='bar',
        stacked=True,
        ax=axs[0,1],
        rot=0
    )
    axs[0,1].set_xlabel('Gender')
    axs[0,1].set_ylabel('Percent of Entire Gender Group')
    axs[0,1].set_title('Overall Gender Percentage per Cluster')
    axs[0,1].legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Gender Percentage by Cluster (within cluster)
    female_vals = cluster_analysis['Percentage Female']
    male_vals = cluster_analysis['Percentage Male']
    other_vals = cluster_analysis['Percentage Other']
    axs[1,1].bar(indices, female_vals, color='lightgreen', label='Female')
    axs[1,1].bar(indices, male_vals, bottom=female_vals, color='lightcoral', label='Male')
    axs[1,1].bar(indices, other_vals, bottom=female_vals + male_vals, color='plum', label='Other')
    axs[1,1].set_title('Gender Percentage by Cluster (within cluster)')
    axs[1,1].legend(title='Gender')
    
    # plot pie chart to show proportion of each cluster
    sizes = cluster_analysis['Number of Students']
    labels = [f'Cluster {c}' for c in clusters]

    axs[2, 1].pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )
    axs[2, 1].set_title('Cluster Proportion of Dataset')
    axs[2, 1].axis('equal')   # ensures the pie is drawn as a circle
    
    # set plot labels and ticks
    for ax in axs.flat:
        if ax is axs[0,1]:
            # gender plot
            ax.set_xticks(np.arange(len(genders)))
            ax.set_xticklabels(genders)
            ax.set_xlabel('Gender')
        elif ax is not axs[2,1]:
            # all the cluster‐indexed plots
            ax.set_xticks(indices)
            ax.set_xticklabels(clusters)
            ax.set_xlabel('Cluster')

        # y‐label logic stays the same
        ax.set_ylabel(
            'Percentage'
            if 'Pct' in ax.get_title() or 'Percentage' in ax.get_title()
            else ax.get_ylabel()
        )

    plt.tight_layout()
    plt.savefig("cluster_analysis.png")
    plt.show()
