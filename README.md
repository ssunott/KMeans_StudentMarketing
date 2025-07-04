# Student Clustering Project

This repository contains an unsupervised clustering pipeline that groups students based on demographic information and interests. We use the K‑Means algorithm, along with methods like the elbow plot, silhouette score and Davis-Bouldin index, to determine the optimal number of clusters.

This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://studentmarketing.streamlit.app/)

---
## Dataset

* Data sourced from Kaggle: https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering 

---

## ⚙️ Preprocessing Steps

1. **Drop empty records** where `gradyear` is missing.
2. **Clean `age`** by removing text suffixes (e.g. "19.Mar") and filling missing ages with median per `gradyear`.
3. **Filter out outliers** in `age` outside the 1st–99th percentile.
4. **Encode `gender`** as numerical value.
5. **Scale** input features.

---

## 📊 Clustering Method

* **Algorithm**: K‑Means
* **Selecting *k***:

  * **Elbow method**: Plot inertia vs. *k* and look for the point of diminishing returns using the Kneed library.
  * **Silhouette score**: Compute average silhouette for *k* = 2…10 and choose the highest.
  * **Davis-Bouldin index**: Compute Davis-Bouldin index and find *k* that produces the minimum index.

---

## 📈 Visualization

* **Elbow plot**: `plot_elbow_score()` generates and saves `{project_root}/elbow_plot.png`.
* **Silhouette plot**: `plot_silhouette_score()` shows silhouette score vs. *k*.
* **Clustering analysis plot**: `plot_cluster()` shows composition of each cluster and top 10 keywords mentioned in each cluster.

---

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

---

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ssunott/KMeans_StudentMarketing.git
   cd KMeans_StudentMarketing

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run streamlit.py

---