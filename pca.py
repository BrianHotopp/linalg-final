import numpy as np
def PCA(data, n_components=2):
    # convert to np array
    data = np.array(data)
    # get the mean
    mean = np.mean(data, axis=0)
    # subtract the mean from each vector in the data
    centered = data - mean
    # calculate the covariance matrix
    covariance_matrix = np.cov(np.transpose(centered))
    # eigenvalue decomposition of covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    # get the indices of the eigenvalues in descending order
    eigen_values_sorted_indices = np.argsort(eigen_values)
    # get the corresponding eigenvectors
    eigen_vectors_sorted = eigen_vectors[:, eigen_values_sorted_indices]
    # get the first n_components eigenvectors
    pca_vecs = eigen_vectors_sorted[:, :n_components]
    # project the centered data onto the first n_components eigenvectors
    projected_data = np.dot(centered, pca_vecs)
    return projected_data