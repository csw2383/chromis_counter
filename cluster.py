import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file_path = '/Users/jennyjin/chromis/C0079_DataOPT2.csv'
data = pd.read_csv(csv_file_path)

# calculate speed and direction
def calculate_speed_and_direction(data):
    data['x_shifted'] = data.groupby('track_id')['x'].shift()
    data['y_shifted'] = data.groupby('track_id')['y'].shift()
    data['frame_shifted'] = data.groupby('track_id')['frame'].shift()

    data['dx'] = data['x'] - data['x_shifted']
    data['dy'] = data['y'] - data['y_shifted']
    data['d_frame'] = data['frame'] - data['frame_shifted']

    data['speed'] = np.sqrt(data['dx']**2 + data['dy']**2) / data['d_frame']
    data['direction'] = np.arctan2(data['dy'], data['dx'])

    return data

data = calculate_speed_and_direction(data)

# encode track_id as a numerical feature
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['track_id_encoded'] = encoder.fit_transform(data['track_id'])

# DBSCAN
def dbscan(X, eps, min_samples):
    labels = np.full(X.shape[0], -1)  
    cluster_id = 0

    def region_query(point_id):
        point = X[point_id]
        distances = np.linalg.norm(X - point, axis=1)
        return np.where(distances < eps)[0]

    def expand_cluster(point_id, neighbors):
        labels[point_id] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            elif labels[neighbor] == 0:
                labels[neighbor] = cluster_id
                new_neighbors = region_query(neighbor)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1

    for point_id in range(X.shape[0]):
        if labels[point_id] == -1:
            neighbors = region_query(point_id)
            if len(neighbors) < min_samples:
                labels[point_id] = 0  
            else:
                cluster_id += 1
                expand_cluster(point_id, neighbors)
    return labels

max_cluster_sizes = []

for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]

    if not frame_data.empty:
        clustering_features = frame_data[['x', 'y', 'speed', 'direction', 'track_id_encoded']].replace([np.inf, -np.inf], np.nan).dropna().values
        
        if clustering_features.size > 0:
            clusters = dbscan(clustering_features, eps=0.1, min_samples=2)
            frame_data = frame_data.iloc[:len(clusters)]
            frame_data['cluster'] = clusters

            largest_cluster_size = frame_data['cluster'].value_counts().max()
            max_cluster_sizes.append(largest_cluster_size)
        else:
            max_cluster_sizes.append(0)
    else:
        max_cluster_sizes.append(0)

plt.figure(figsize=(14, 7))
plt.bar(sorted(data['frame'].unique()), max_cluster_sizes)
plt.xlabel('Frame Number')
plt.ylabel('Number of Fish in Largest Cluster')
plt.title('Number of Fish in Largest Cluster per Frame')
plt.grid(True)
plt.show()
