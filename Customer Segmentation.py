# %%
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# %%
DATABASE_URL = "mysql+pymysql://root:Dhruv001@localhost/csv_sql"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# %%
# RFM
query = '''SELECT
        CustomerID,
        MAX(OrderDate) AS last_purchase_date,
        DATEDIFF(CURDATE(), MAX(OrderDate)) AS recency,
        COUNT(OrderID) AS frequency,
        SUM(Sales) AS monetary
        FROM DATA
        GROUP BY CustomerID
        '''
df = pd.read_sql(query, engine)

session.close()

# %%
# Removing the top 2% of spenders if they are outliers
df = df[df['monetary'] < df['monetary'].quantile(0.98)]

# Scale the features (recency, frequency, monetary)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['recency', 'frequency', 'monetary']])

# Convert the scaled features back into a DataFrame (optional)
df_scaled = pd.DataFrame(df_scaled, columns=['recency', 'frequency', 'monetary'])

# %%
# Function to calculate the within-cluster sum of squares (WCSS)
def calculate_wcss(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    return kmeans.inertia_

# %%
# Function to compute the gap statistic
def gap_statistic(X, k_max, wcss, B=100):
    gap_values = []
    
    for k in range(1, k_max + 1):
        # Calculate WCSS for the actual data
        W_k = wcss[k-1]
        
        # Generate random uniform data
        random_data = np.random.rand(*X.shape)
        
        # Calculate WCSS for the random data (B times)
        W_k_b = np.zeros(B)
        for b in range(B):
            W_k_b[b] = calculate_wcss(random_data, k)
        
        # Calculate gap statistic
        gap = np.log(np.mean(W_k_b)) - np.log(W_k)
        gap_values.append(gap)
    
    print(W_k)

    return gap_values

# %%
# Elbow Method to find the optimal 'k'
wcss = []  # List to hold WCSS for each value of 'k'
silhouette_scores = []
gap_values = []
calinski_harabasz_scores = []
k_range = range(1, 11)

# %%
for i in k_range:  # Trying k values from 1 to 10
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    # Elbow method
    wcss.append(kmeans.inertia_)  # WCSS is the sum of squared distances to centroids
    # Calculate the silhouette score for the current clustering
    if i == 1:
        continue
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    # Calculate the Calinski-Harabasz index for the current clustering
    calinski_score = calinski_harabasz_score(df_scaled, kmeans.labels_)
    calinski_harabasz_scores.append(calinski_score)
print(wcss)
# Calculate gap statistic for different k values
gap_values = gap_statistic(df_scaled, 10, wcss)

# Calculate differences (delta^2 WCSS)
diff = np.diff(np.diff(wcss))

# %%
# Plot WCSS for the elbow method
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1, 11)), 
    y=wcss,
    mode='lines+markers', 
    name='WCSS',
    line=dict(color='blue', width=3),
    marker=dict(color='red', size=8)
))

fig.update_layout(
    title='Elbow Method for Optimal Number of Clusters',
    xaxis=dict(title='Number of Clusters (k)'),
    yaxis=dict(title='WCSS (Within-Cluster Sum of Squares)'),
    template='plotly_dark'
)

fig.show()

# %%
# Create the plot for silhouette scores using Plotly Graph Objects
fig = go.Figure()

# Add a line plot for silhouette scores
fig.add_trace(go.Scatter(x=list(range(2, 11)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score',line=dict(color='blue', width=3),marker=dict(color='red', size=8)))

# Add title and labels
fig.update_layout(
    title="Silhouette Score vs. Number of Clusters",
    xaxis_title="Number of Clusters",
    yaxis_title="Silhouette Score",
    template="plotly_dark"  # Optional: set dark theme for better visuals
)

# Show the plot
fig.show()

# %%
# Plot the gap statistic
fig_gap = go.Figure()

fig_gap.add_trace(go.Scatter(
    x=list(range(1, 11)), 
    y=gap_values,
    mode='lines+markers', 
    name='Gap Statistic',
    line=dict(color='green', width=3),
    marker=dict(color='orange', size=8)
))

fig_gap.update_layout(
    title='Gap Statistic vs. Number of Clusters',
    xaxis=dict(title='Number of Clusters (k)'),
    yaxis=dict(title='Gap Statistic'),
    template='plotly_dark'
)

fig_gap.show()

# %%
# Plot the Calinski-Harabasz Index
fig_calinski = go.Figure()

fig_calinski.add_trace(go.Scatter(
    x=list(range(2, 11)), 
    y=calinski_harabasz_scores,
    mode='lines+markers', 
    name='Calinski-Harabasz Index',
    line=dict(color='purple', width=3),
    marker=dict(color='yellow', size=8)
))

fig_calinski.update_layout(
    title='Calinski-Harabasz Index vs. Number of Clusters',
    xaxis=dict(title='Number of Clusters (k)'),
    yaxis=dict(title='Calinski-Harabasz Index'),
    template='plotly_dark'
)

fig_calinski.show()

# %%
k_elbow = np.argmax(diff) + 2  # +2 because the second difference is for (k-1)th
print(f"Optimal number of clusters according to Elbow Method: {k_elbow}")
k_silhouette = k_range[np.argmax(silhouette_scores)+1]  # Adding 1 because index starts at 0
print(f"Optimal number of clusters according to Silhouette Score: {k_silhouette}")
k_gap = np.argmax(gap_values) + 1  # Adding 1 because index starts at 0
print(f"Optimal number of clusters according to Gap Statistic: {k_gap}")
k_calinski = np.argmax(calinski_harabasz_scores) + 2  # Adding 2 because the first index is for k=1
print(f"Optimal number of clusters according to Calinski-Harabasz Index: {k_calinski}")

# %%
while(True):
    k = int(input("Enter number of clusters for k-means clustering: "))
    if(k>0 and k<11):
        break
    else:
        print("Invalid number entered.")

# %%
# Perform KMeans clustering with the chosen k value
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Apply PCA for dimensionality reduction to 2D (for visualization)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# %%
# Plot the customer segmentation clusters using Plotly in the PCA space
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_pca[:, 0], 
    y=df_pca[:, 1], 
    mode='markers', 
    marker=dict(
        color=df['cluster'], 
        colorscale='Viridis', 
        size=12, 
        opacity=0.6, 
        line=dict(width=1)
    ),
    text=df['CustomerID'],  # Optional: Show CustomerID on hover
    name='Customer Segments'
))

# Cluster centroids in the PCA space
centroids = pca.transform(kmeans.cluster_centers_)

fig.add_trace(go.Scatter(
    x=centroids[:, 0], 
    y=centroids[:, 1], 
    mode='markers', 
    marker=dict(
        color='red', 
        size=12, 
        symbol='x'
    ),
    name='Centroids'
))

fig.update_layout(
    title='Customer Segmentation - PCA of Recency, Frequency, and Monetary',
    xaxis=dict(title='PCA Component 1'),
    yaxis=dict(title='PCA Component 2'),
    template='plotly_dark',
    hovermode='closest'
)

fig.show()

# %%
# Output the loadings (coefficients) of the principal components
loadings = pca.components_

# Create a DataFrame to make it easier to view
loadings_df = pd.DataFrame(loadings, columns=['Recency', 'Frequency', 'Monetary'], index=['PC1', 'PC2'])

# Display the loadings
print("Linear impact coefficients of different metrics on the plotted components:")
print(loadings_df)