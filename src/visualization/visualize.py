
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

def plot_feature_importance(model, x):
    """
    Plot a bar chart showing the top 5 feature importances.
    
    Args:
        feature_names (list): List of feature names.
        feature_importances (list): List of feature importance values.
    """
   # 1. Create a DataFrame of feature importances
    feat_imp = pd.DataFrame({
        'Feature': x.columns,
        'Importance': model.feature_importances_
    })

    # 2. Extract generic feature name from encoded feature (e.g., 'Manufacturer_BMW' vs 'Manufacturer')
    feat_imp['Group'] = feat_imp['Feature'].str.extract(r'^([^_]+)')

    # 3. Aggregate by group
    grouped_imp = feat_imp.groupby('Group')['Importance'].sum().reset_index()

    # 4. Sort
    grouped_imp = grouped_imp.sort_values(by='Importance', ascending=False)

    # 5. Plot
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Group', data=grouped_imp, ax=ax)
    plt.title("Features by Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    fig.savefig("feature_importance.png")


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix for the given true and predicted labels.
    
    Args:
        y_true (numpy.ndarray): Array of true labels.
        y_pred (numpy.ndarray): Array of predicted labels.
        classes (list): List of class labels.
        normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
        title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=16)
    plt.show()