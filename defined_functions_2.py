import numpy as np

def find_covariance(dataFrame):
    a = dataFrame[0]
    b = dataFrame[1]
    c = dataFrame[2]

    mean_a = sum(a) / len(dataFrame[0])
    mean_b = sum(b) / len(dataFrame[1])
    mean_c = sum(c) / len(dataFrame[2])

    cov_matrix = np.zeros((3, 3))

    for i in range(len(dataFrame[0])):
        cov_matrix[0, 0] += (a[i] - mean_a)**2
        cov_matrix[1, 1] += (b[i] - mean_b)**2
        cov_matrix[2, 2] += (c[i] - mean_c)**2
        cov_matrix[0, 1] += (a[i] - mean_a) * (b[i] - mean_b)
        cov_matrix[0, 2] += (a[i] - mean_a) * (c[i] - mean_c)
        cov_matrix[1, 2] += (b[i] - mean_b) * (c[i] - mean_c)

    cov_matrix[0, 1] /= len(dataFrame[0])
    cov_matrix[0, 2] /= len(dataFrame[0])
    cov_matrix[1, 2] /= len(dataFrame[0])
    cov_matrix[1, 0] = cov_matrix[0, 1]
    cov_matrix[2, 0] = cov_matrix[0, 2]
    cov_matrix[2, 1] = cov_matrix[1, 2]

    cov_matrix[0, 0] /= len(dataFrame[0])
    cov_matrix[1, 1] /= len(dataFrame[0])
    cov_matrix[2, 2] /= len(dataFrame[0])

    return(cov_matrix)

def Standard_least_Square(a, b, c):
    #adding a ones row for 3x3 operation
    A = np.column_stack((a, b, np.ones_like(a)))

    # Define the vector b
    B = np.row_stack(c)

    # Compute the normal equations and obtain coefficients
    a, b, c  = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, B))

    print("Coefficients of Least Square plane equation are: ", a, b, c, "\n")

    x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    z = a * x + b * y + c

    return x, y, z

def Total_Least_Square(dataFrame):
    A_mean = np.mean(dataFrame, axis=0)
    A_centered = dataFrame - A_mean

    # Compute the covariance matrix of the centered data matrix
    cov = np.dot(A_centered.T, A_centered)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)

    # Extract the eigenvector corresponding to the smallest eigenvalue
    n = eigvecs[:, np.argmin(eigvals)]

    # Compute the optimized coefficients
    a, b, c = n[:3]
    d = -(a*A_mean[0] + b*A_mean[1] + c*A_mean[2])

    # Print the coefficients
    print("Coefficients of Total Least Square plane equation are: ", a, b, c, d, "\n")

    #generate data to plot
    x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    z = (-a * x + -b * y + -d) * 1/c

    return x, y, z

def RANSAC_fit_surface(points, max_iterations=1000, threshold=0.2):
    optimum_fit = None
    best_inliers = None
    best_inlier_count = 0
    
    for i in range(max_iterations):
        # Select 3 random indices to fit a plane
        sample_indices = np.random.randint(0, points.shape[0], 3)
        
        # Get the corresponding points
        p1, p2, p3 = points[sample_indices]
        
        # Fit a plane using 3 points
        normal = np.cross(p2 - p1, p3 - p1)
        d = -np.dot(normal, p1)
        
        # Calculate the distances between the plane and all other points
        distances = np.abs(normal[0] * points[:,0] + 
                           normal[1] * points[:,1] + 
                           normal[2] * points[:,2] + d) / np.linalg.norm(normal)
        
        # Find the inliers that have a distance less than the threshold
        inliers = np.where(distances < threshold)[0]
        inlier_count = len(inliers)
        
        # If we have found a better model, update the best model
        if inlier_count > best_inlier_count:
            optimum_fit = (normal, d)
            best_inliers = inliers
            best_inlier_count = inlier_count
    
    return optimum_fit

