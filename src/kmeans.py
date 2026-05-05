import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

data.head()

X = data[['Annual Income (k$)','Spending Score (1-100)']].values
kmeans = KMeans(n_clusters=5, random_state=42)
y_means = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_

plt.scatter(centroids[:,0], centroids[:,1], 
            s=200, c='black', marker='X')
