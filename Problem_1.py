import cv2
import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from defined_functions_1 import load_images
from defined_functions_1 import calculate_centroid
from defined_functions_1 import least_square_fit_curve
from defined_functions_1 import plot_curve
from defined_functions_1 import get_centroids
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- PROBLEM 1 - Question 1 - Ball Tracking----------------------------------------------
#load video
vid = cv2.VideoCapture('ball.mov')

#captures all frames from vid
video_frames = load_images(vid)

#generate red object filter
mask = cv2.inRange(video_frames[100], np.array([0, 0, 130]),  np.array([65, 65, 255]))
output = cv2.bitwise_and(video_frames[100], video_frames[100], mask = mask)

#taking one of the frames as sample
img_copy = video_frames[1]
gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,0,50,0)

#Mark centroid on ball for a first random image
try:
  cX, cY = calculate_centroid(thresh)
  cv2.circle(thresh, (cX, cY), 5, (255, 255, 255), -1)
  cv2.putText(thresh, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
  ball_found = cv2.bitwise_and(video_frames[100], video_frames[100], mask = thresh)
except:
  None

#generating figure with ball extraction
fig1 = plt.figure("Problem 1.1")
fig1.add_subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(video_frames[100], cv2.COLOR_BGR2RGB))
fig1.add_subplot(2, 1, 2)
plt.imshow(ball_found)

#Extract x,y coordinate of ball from each frame of video
coordinates = get_centroids(video_frames)

#figure representing the least square fitting done on graph
fig2 = plt.figure("Standard Least Square fitting")
implot = plt.imshow(cv2.cvtColor(video_frames[100], cv2.COLOR_BGR2RGB))

# x and y coordinates of all the detected ball in frames
x = [i[0] for i in coordinates]
y = [i[1] for i in coordinates]

x_data = np.array(x)
y_data = np.array(y)

# ----------------------------- PROBLEM 1 - Question 2 and 3 - Standard Least Square----------------------------------------------

#this function calculates and gives the coefficients of the parabolic equation
coefficients = least_square_fit_curve(x_data, y_data)
[a, b, c] = coefficients

print('Equation of the fitted curve is: \n')
print('y = (%f)x^2 + (%f)x + %f' %(a,b,c), "\n")

#making changes to c value as asked in part 3
y_initial = c
y_final = y_initial + 300
c = y_initial - y_final

# find two roots for specific c value
ans1 = (-b-cmath.sqrt((b**2) - (4 * a*c)))/(2 * a)
ans2 = (-b+cmath.sqrt((b**2) - (4 * a*c)))/(2 * a)

# printing the roots
print('The roots are ', ans1, " and ",ans2, "\n")

# In this case, x is positive, hence we will consider our positive value
print("X coordinate of Landing spot is ", ans2, "\n")

#this function plots the values
plot_curve(x_data, y_data, coefficients)