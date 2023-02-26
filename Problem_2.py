import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from defined_functions_2 import find_covariance
from defined_functions_2 import Standard_least_Square
from defined_functions_2 import Total_Least_Square
from defined_functions_2 import RANSAC_fit_surface
warnings.filterwarnings('ignore')

# Load data from a CSV file with no header
dataFrame1 = np.transpose(np.array(pd.read_csv("pc1.csv", header=None)))
a1,b1,c1 = dataFrame1

dataFrame2 = np.transpose(np.array(pd.read_csv("pc2.csv", header=None)))
a2,b2,c2 = dataFrame2

#Problem 2. 1. a.
cov_matrix = find_covariance(dataFrame1)
print("The covariance matrix for PC1 is: \n", cov_matrix, "\n")

# -------------- Problem 2. 1. b. ----------------------
eigen_values, eigen_vector = np.linalg.eig(cov_matrix)
print("The eigen values are as follows: ", eigen_values, "\n")
print("The eigen vector is as follows:\n ",eigen_vector, "\n")

# we know that min eigen value depicts surface normal
normal_direction = eigen_vector[:, np.argmin(eigen_values)]
print("The direction of normal vector is: ", normal_direction,"\n")
print("The magnitude of the normal is: ", np.sqrt(eigen_values[np.argmin(eigen_values)]), "\n")
#-------------------------------------------------------

# -------------- Problem 2. 2. a. ----------------------
# Standard Least Square for PC1
fig = plt.figure("Standard Least Square")
fig.suptitle("Standard Least Square Plots")
first_subplot = fig.add_subplot(1, 2, 1, projection='3d')
first_subplot.set_title("SLS for PC1")
x1, y1, z1 = Standard_least_Square(a1,b1,c1)
first_subplot.scatter(a1, b1, c1, c='r', marker='o', alpha=0.3)
first_subplot.plot_surface(x1,y1,z1,color="yellow")

# Standard Least Square for PC2
second_subplot = fig.add_subplot(1, 2, 2, projection='3d')
second_subplot.set_title("SLS for PC2")
x2, y2, z2 = Standard_least_Square(a2,b2,c2)
second_subplot.scatter(a2, b2, c2, c='r', marker='o', alpha=0.3)
second_subplot.plot_surface(x2,y2,z2,color="yellow")

# Total Least Square for PC1
fig1 = plt.figure("Total Least Square")
fig1.suptitle("Total Least Square Plots")
first_subplot = fig1.add_subplot(1, 2, 1, projection='3d')
first_subplot.set_title("TLS for PC1")
x1, y1, z1 = Total_Least_Square(dataFrame1.T)
first_subplot.scatter(a1, b1, c1, c='r', marker='o', alpha=0.3)
first_subplot.plot_surface(x1, y1, z1, color="yellow")

# Total Least Square for PC2
second_subplot = fig1.add_subplot(1, 2, 2, projection='3d')
second_subplot.set_title("TLS for PC2")
x2, y2, z2 = Total_Least_Square(dataFrame2.T)
second_subplot.scatter(a2, b2, c2, c='r', marker='o', alpha=0.3)
second_subplot.plot_surface(x2, y2, z2, color="yellow")
#-------------------------------------------------------

# -------------- Problem 2. 2. b. ----------------------
#---------------- RANSAC PC1 -----------------
surface = RANSAC_fit_surface(dataFrame1.T)

#Extracting surface coefficients
(a,b,c), d = surface

print("RANSAC Surface equation coefficients for PC1 are: a =", a, " b =", b, "c =", c, "d =", d, "\n")

x1, y1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z1 = (-a * x1 - b * y1 - d) * 1 / c

fig3 = plt.figure("RANSAC")
fig3.suptitle("RANSAC Plots")
first_subplot = fig3.add_subplot(1,2,1, projection='3d')
first_subplot.set_title("RANSAC for PC1")
first_subplot.scatter(a1, b1, c1, c='r', marker='o', alpha=0.3)
first_subplot.plot_surface(x1, y1, z1, color="yellow", alpha = 0.6)

#---------------- RANSAC PC2 -----------------
surface = RANSAC_fit_surface(dataFrame2.T)

#Extracting surface coefficients
(a,b,c), d = surface

print("RANSAC Surface equation coefficients for PC2 are: a =", a, " b =", b, "c =", c, "d =", d, "\n")

second_subplot = fig3.add_subplot(1,2,2, projection='3d')
second_subplot.set_title("RANSAC for PC2")

x2, y2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z2 = (-a * x2 - b * y2 - d) * 1 / c

second_subplot.scatter(a2, b2, c2, c='r', marker='o', alpha=0.3)
second_subplot.plot_surface(x2, y2, z2, color="yellow", alpha = 0.6)
#-------------------------------------------------------

plt.show()