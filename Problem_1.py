import cv2
import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings

warnings.filterwarnings('ignore')

def calculate_centroid(image):
  y_values, x_values = np.nonzero(image)
  X = np.sum(x_values)/x_values.shape
  Y = np.sum(y_values)/y_values.shape
  return (int(X), int(Y))

vid = cv2.VideoCapture('ball.mov')
img = cv2.imread('ball_snap.jpg')

lower_red_threshold = np.array([0, 0, 130], dtype = "uint8") 
upper_red_threshold= np.array([65, 65, 255], dtype = "uint8")
filtered = cv2.inRange(img, lower_red_threshold, upper_red_threshold)
output = cv2.bitwise_and(img, img, mask = filtered)

cv2.imshow(" ", output)

i=0
cv_img = []
success, frame = vid.read()
success = True
while success:
    success, frame = vid.read()
    if success == True:
      cv_img.append(frame) 

img_copy = cv_img[1]
gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,0,50,0)

print("Number of images from Video feed: ", len(cv_img), "\n")

#Mark centroid on ball
try:
  cX, cY = calculate_centroid(thresh)
  cv2.circle(thresh, (cX, cY), 5, (255, 255, 255), -1)
  cv2.putText(thresh, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
except:
  None

#Extract x,y coordinate of ball from each frame of video
coordinates = []
for images in cv_img[:]:
    filtered = cv2.inRange(images, lower_red_threshold, upper_red_threshold)
    output = cv2.bitwise_and(images, images, mask = filtered) 
    gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,0,255,0)

    try:
      cX, cY = calculate_centroid(thresh)
      cv2.circle(images, (cX, cY), 5, (255, 255, 255), -1)
      specific_coordinate = (cX, cY)
      coordinates.append(specific_coordinate)
    except:
      continue

im = plt.imread('ball_snap.jpg')
implot = plt.imshow(im)

#x and y coordinates of all the detected ball in frames
x = [i[0] for i in coordinates]
y = [i[1] for i in coordinates]

#Extracting ball coordinates from every 6th frame for easy visulization
x = [x[i] for i in range(0, len(x), 6)]
y = [y[i] for i in range(0, len(y), 6)]

#Scatter plot for visualization of selected coordinates from video
plt.scatter(x, y, s=10, marker='o', color="red")

# ----------------------------- PROBLEM 1 - Question 2 - Standard Least Square----------------------------------------------

#Function to calculate coefficient values for the given x and y pair of values
def least_square_fit_curve(x_data, y_data, degree):
    X = np.column_stack([x_data**i for i in range(degree+1)])

    X_T = X.transpose()
    y = y_data.reshape((-1, 1))

    coefficients = np.linalg.inv(X_T @ X) @ X_T @y
    coefficients = coefficients.flatten()

    return coefficients

#This function plots the curves and makes use of coefficients and given x and y values
def plot_curve(x_data, y_data, coefficients):
    # Get the range of x values for plotting
    im_a = plt.imread('ball_Moment.jpg')
    im_a = plt.imshow(im_a)
    
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_vals = np.linspace(x_min, x_max, 100)

    # Evaluate the fitted curve at the x values
    y_vals = np.zeros(x_vals.size)
    for i, x in enumerate(x_vals):
        for j, c in enumerate(coefficients):
            y_vals[i] += c * x**j

    # Plot the data points and the fitted curve
    plt.plot(x_data, y_data, 'bo', label='Data points')
    # plt.plot(x_vals, y_vals, 'r', label='Fitted curve')
    plt.legend()
    plt.show()

x_data = np.array(x)
y_data = np.array(y)

# Fit the curve to the data
coefficients = least_square_fit_curve(x_data, y_data, 2)

c = 456.023218
b = -0.599493
a = 0.000591
coefficients = [c, b, a]

print('Equation of the fitted curve is: \n')
print('y = %f + (%f)x + (%f)x^2' %(c,b,a), "\n")

plot_curve(x_data, y_data, coefficients)

y1 = c+300

#converting the equation y1 = ax^2 + bx + c into 0 = ax^2 + bx + c, so it becomes quadratic, we will calculate c as c-y1
c = c - y1

# find two roots
ans1 = (-b-cmath.sqrt((b**2) - (4 * a*c)))/(2 * a)
ans2 = (-b+cmath.sqrt((b**2) - (4 * a*c)))/(2 * a)
  
# printing the roots
print('The roots are ', ans1, " and ",ans2, "\n")

#In this case, x is positive, hence as consider our positive value
print("X coordinate of Landing spot is ", ans2, "\n")
