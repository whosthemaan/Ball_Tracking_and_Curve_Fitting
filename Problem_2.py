import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings('ignore')

# Load data from a CSV file and hide the header
dataFrame1 = pd.read_csv("pc1.csv", header=None)
dataFrame1 = np.array(dataFrame1)
dataFrame1 = np.transpose(dataFrame1)

a1 = dataFrame1[0]
b1 = dataFrame1[1]
c1 = dataFrame1[2]

dataFrame2 = pd.read_csv("pc2.csv", header=None)
dataFrame2 = np.array(dataFrame2)
dataFrame2 = np.transpose(dataFrame2)

a2 = dataFrame2[0]
b2 = dataFrame2[1]
c2 = dataFrame2[2]

def find_covariance(dataFrame):
    a = dataFrame[0]
    b = dataFrame[1]
    c = dataFrame[2]

    n = 300

    mean_a = sum(a) / n
    mean_b = sum(b) / n
    mean_c = sum(c) / n

    cov_matrix = np.zeros((3, 3))

    for i in range(n):
        cov_matrix[0, 0] += (a[i] - mean_a)**2
        cov_matrix[1, 1] += (b[i] - mean_b)**2
        cov_matrix[2, 2] += (c[i] - mean_c)**2
        cov_matrix[0, 1] += (a[i] - mean_a) * (b[i] - mean_b)
        cov_matrix[0, 2] += (a[i] - mean_a) * (c[i] - mean_c)
        cov_matrix[1, 2] += (b[i] - mean_b) * (c[i] - mean_c)

    cov_matrix[0, 1] /= n
    cov_matrix[0, 2] /= n
    cov_matrix[1, 2] /= n
    cov_matrix[1, 0] = cov_matrix[0, 1]
    cov_matrix[2, 0] = cov_matrix[0, 2]
    cov_matrix[2, 1] = cov_matrix[1, 2]

    cov_matrix[0, 0] /= n
    cov_matrix[1, 1] /= n
    cov_matrix[2, 2] /= n

    return(cov_matrix)

cov_matrix = find_covariance(dataFrame1)
print("The covariance matrix for PC1 is: \n", cov_matrix, "\n")

# Problem 2. 1. b.
eigen_values, eigen_vector = np.linalg.eig(cov_matrix)
print("The eigen values are as follows: ", eigen_values, "\n")
print("The eigen vector is as follows:\n ",eigen_vector,"\n")

# we know that min eigen value depicts the direction of normal_direction
normal_direction = eigen_vector[:, np.argmin(eigen_values)]
print("The direction of normal vector is: ", normal_direction,"\n")
print("The magnitude of the normal is: ", eigen_values[np.argmin(eigen_values)], "\n")

# Problem 2. 2. a.
def Standard_least_Square(a, b, c):
    A = np.column_stack((a, b, np.ones_like(a)))

    # Define the vector b
    b = np.row_stack(c)

    # Compute the normal equations
    ATA = np.dot(A.T, A)
    ATAInv = np.linalg.inv(ATA)
    ATb = np.dot(A.T, b)

    x = np.dot(ATAInv, ATb)

    # Extract the optimized coefficients
    a, b, c = x

    print("Coefficients of Least Square plane equation are: ", a, b, c, "\n")

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    x, y = np.meshgrid(x, y)
    z = a * x + b * y + c

    return x, y, z

# Standard Least Square for PC1
fig = plt.figure("Standard Least Square")
fig.suptitle("Standard Least Square Plots")
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title("SLS for PC1")
x1, y1, z1 = Standard_least_Square(a1,b1,c1)
ax.plot_surface(x1,y1,z1,cmap=cm.coolwarm)

# Standard Least Square for PC2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title("SLS for PC2")
x2, y2, z2 = Standard_least_Square(a2,b2,c2)
ax2.plot_surface(x2,y2,z2,cmap=cm.coolwarm)
# plt.show()

def Total_Least_Square(a,b,c):
    A = np.column_stack((a, b, c))
    A_mean = np.mean(A, axis=0)
    A_centered = A - A_mean

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

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    x, y = np.meshgrid(x, y)
    z = (-a/c) * x + (-b/c) * y + (-d/c)

    return x, y, z

# Total Least Square for PC1
fig1 = plt.figure("Total Least Square")
fig1.suptitle("Total Least Square Plots")
ax = fig1.add_subplot(1, 2, 1, projection='3d')
ax.set_title("TLS for PC1")
x1, y1, z1 = Total_Least_Square(a1,b1,c1)
ax.plot_surface(x1,y1,z1,cmap=cm.coolwarm)

# Total Least Square for PC2
ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
ax2.set_title("TLS for PC2")
x2, y2, z2 = Total_Least_Square(a2,b2,c2)
ax2.plot_surface(x2,y2,z2,cmap=cm.coolwarm)
# plt.show()

# Problem 2. 2. b.
points1 = np.column_stack((a1,b1,c1))
points2 = np.column_stack((a2,b2,c2))

def RANSAC_fit_surface(points, max_iterations=1000, threshold=0.1):
    best_model = None
    best_inliers = None
    best_inlier_count = 0
    num_points = points.shape[0]
    
    for i in range(max_iterations):
        # Select 3 random indices to fit a plane
        sample_indices = np.random.randint(0, num_points, 3)
        
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
            best_model = (normal, d)
            best_inliers = inliers
            best_inlier_count = inlier_count
    
    return best_model, best_inliers

#------------------------------- RANSAC PC1 ------------------------------------------
surface, inliers = RANSAC_fit_surface(points1)

#Extracting surface coefficients
a = surface[0][0]
b = surface[0][1]
c = surface[0][2]
d = surface[1]

print("Surface equation coefficients for PC1 are: a =", a, " b =", b, "c =", c, "d =", d, "\n")

fig3 = plt.figure("RANSAC")
ax = fig3.add_subplot(1,2,1, projection='3d')
ax.set_title("RANSAC for PC1")

# Plot all points
# ax.scatter(points1[:,0], points1[:,1], points1[:,2], c='b', marker='o')
# ax.scatter(points1[inliers,0], points1[inliers,1], points1[inliers,2], c='r', marker='o')

normal, d = surface
xx, yy = np.meshgrid(range(2), range(2))
zz = (-a * xx - b * yy - d) * 1. / c

#Plotting surface
ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm)

# ------------------------------- RANSAC PC2 ---------------------------------------
surface, inliers = RANSAC_fit_surface(points2)

#Extracting surface coefficients
a = surface[0][0]
b = surface[0][1]
c = surface[0][2]
d = surface[1]

print("Surface equation coefficients for PC2 are: a =", a, " b =", b, "c =", c, "d =", d, "\n")

ax1 = fig3.add_subplot(1,2,2, projection='3d')
ax1.set_title("RANSAC for PC2")

# Plot all points
# ax1.scatter(points1[:,0], points1[:,1], points1[:,2], c='b', marker='o')
# ax1.scatter(points1[inliers,0], points1[inliers,1], points1[inliers,2], c='r', marker='o')

normal, d = surface
xx, yy = np.meshgrid(range(2), range(2))
zz = (-a * xx - b * yy - d) * 1. / c

#Plotting surface
ax1.plot_surface(xx, yy, zz, cmap=cm.coolwarm)

plt.show()

