from sklearn.cluster import DBSCAN
import numpy as np

def filter_solutions_2d_with_error(points, eps=0.1, min_samples=2):
    X = np.array(points[:, :2])  # Extract only the x and y coordinates for clustering
    
    # Use DBSCAN to cluster the points
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    print(labels)
    # Extract clusters
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((points[i, 0], points[i, 1], points[i, 2]))

    # Filter out noise points (not assigned to any cluster)
    clusters = {k: v for k, v in clusters.items() if k != -1}

    # Keep the point with the least error for each cluster
    cluster_representatives = {}
    for label, cluster_points in clusters.items():
        min_error_point = min(cluster_points, key=lambda x: x[2])
        cluster_representatives[label] = (min_error_point[0], min_error_point[1])

    return cluster_representatives

# Example usage
points_2d_with_error = np.array([
    [1.989298191, 5.011210959, 0.001939448],
    [2.013759435, 4.988990179, 0.006371717],
    [2.021196011, 5.011247344, 0.000523618],
    [2.002984323, 5.015239919, 0.008766798],
    [2.017880607, 4.986871799, 0.000319496],
    [2.01934037, 5.000678774, 0.00954417],
    [2.018041543, 5.007424053, 0.007818004],
    [1.995406748, 5.016740004, 0.003482161],
    [2.006776914, 4.983919561, 0.009216941],
    [2.024287864, 4.986902156, 0.007563012],
    [2.012669668, 5.011891514, 0.004851955],
    [1.987714322, 4.975989415, 0.006382611],
    [1.988804852, 4.999192764, 0.008944196],
    [3.010751478, -0.988466739, 0.002455872],
    [3.009824703, -1.01735109, 0.003516233],
    [3.009138864, -1.008428189, 0.001478755],
    [3.00786224, -0.994145172, 0.002598764],
    [2.997672633, -0.992611741, 0.00253171],
    [3.010160503, -0.992298861, 0.00188673],
    [2.983060689, -0.992418511, 0.009794737],
    [3.008791874, -0.989873828, 0.005052704],
    [3.019297987, -0.996135435, 0.008329732],
    [3.01282744, -1.015445536, 0.007960832],
    [2.975943224, -1.022835899, 0.002256471],
    [3.000741718, -1.024995312, 0.000696151],
    [3.017802227, -1.014501984, 0.00673929],
    [7.008196903, 4.005686846, 0.002829453],
    [6.984150169, 4.022027041, 0.007169176],
    [7.019631138, 4.011113345, 0.008941117],
    [7.020482674, 4.024026676, 0.008933171],
    [7.013254112, 4.000767918, 0.006898311],
    [6.977136108, 3.99355473, 0.009270366],
    [7.023021028, 3.99699439, 0.00991223],
    [7.016954589, 4.010305751, 0.006788674],
    [6.979799809, 3.984179277, 0.006500637],
    [6.994906955, 4.011459065, 0.005707023],
    [7.015204428, 3.987288841, 0.008259478],
    [7.006205015, 3.980717262, 0.009293204],
    [7.002816939, 3.981439233, 0.00399175]
])

cluster_representatives_with_error = filter_solutions_2d_with_error(points_2d_with_error)

# Print cluster representatives
for label, representative in cluster_representatives_with_error.items():
    print(f"Cluster {label} Representative: {representative}")
