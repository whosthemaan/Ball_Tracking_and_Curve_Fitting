<!-- The steps to run the following code involves two very basic steps: -->
<!-- All the libraries used in my code are standard libraries including numpy, matplotlib, pandas and cv2 -->
## Ball tracking in a video and curve fitting using Standard Least Square Method

### Problem 1:
In the given video, a red ball is thrown against a wall. Assuming that the trajectory of the ball follows
the equation of a parabola:

1. Detect and plot the pixel coordinates of the center point of the ball in the video.
The below image demonstrates one of the snap out of the video
<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221436247-a8ac8cde-3551-4a5a-bd71-6e841a9a3007.png" alt= “” width="400" height="350">
</p>
<p align="center"> Fig1. Ball color identification </p>

&nbsp;&nbsp; 2. Use Standard Least Squares to fit a curve to the extracted coordinates. For the estimated parabola you must,

&nbsp;&nbsp;&nbsp;&nbsp; a. Equation of the curve is: **y = 456.02318 + (-0.599493)x + (0.000591)x^2**

&nbsp;&nbsp;&nbsp;&nbsp; b. Plot the data with best fit curve

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221065980-14cb6029-8bd3-4fbc-9a0d-b4467a7d0ba3.png" alt= “” width="400" height="350">
<p>

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221063745-417a8ae1-cd70-4600-b510-e9699ddfef25.png" alt= “” width="400" height="350">
</p>

3. Assuming that the origin of the video is at the top-left of the frame as shown below, compute the x-coordinate of the ball’s landing spot in pixels, if the y-coordinate of the landing spot is defined as 300 pixels greater than its first detected location.

&nbsp;&nbsp;&nbsp;&nbsp; With the above equation, we know that at X = 0, Y is 456, therefore according to question, we will get Y as 456 + 300 = 756. Using the value of Y, we have to find the value of x and hence solve the quadratic equation. 

 &nbsp;&nbsp;&nbsp;&nbsp; The roots of x are as follows: **(-367.37)** and **(1381.74)**

 &nbsp;&nbsp;&nbsp;&nbsp; <b>We will consider the positive value and hence the x value to be considered is (1381.74)</b>

### Problem 2:

Given are two csv files, pc1.csv and pc2.csv, which contain noisy LIDAR point cloud data in the form of (x, y, z) coordinates of the ground plane.

### Plot from PC1

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221076723-288475f5-ef0e-4f3f-9817-0b9edaa67e5f.png" alt= “PC1” width="500" height="350">
</p>

### Plot from PC2

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221076784-d20d93d9-f671-4ad4-9374-6307e3a5c6bc.png" alt= “PC2” width="500" height="350">
</p>

&nbsp;&nbsp; 1. Using pc1.csv:

&nbsp;&nbsp;&nbsp;&nbsp; a. Compute the covariance matrix

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The covariance matrix for PC1 is: 
 [[ 33.6375584   -0.82238647 -11.3563684 ]
 [ -0.82238647  35.07487427 -23.15827057]
 [-11.3563684  -23.15827057  20.5588948 ]] 
 
 &nbsp;&nbsp;&nbsp;&nbsp; b. Assuming that the ground plane is flat, use the covariance matrix to compute the magnitude and direction of the surface normal.
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The eigen values are as follows:  [ 0.66727808 34.54205318 54.06199622] 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The eigen vector is as follows:
  [[ 0.28616428  0.90682723 -0.30947435]
 [ 0.53971234 -0.41941949 -0.72993005]
 [ 0.79172003 -0.04185278  0.60944872]] 
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The direction of normal vector is:  [0.28616428 0.53971234 0.79172003] 
 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The magnitude of the normal is:  0.8168709081065844 
  
&nbsp;&nbsp; 1. In this question, you will be required to implement various estimation algorithms such as Standard Least Squares, Total Least Squares and RANSAC:

&nbsp;&nbsp;&nbsp;&nbsp; a. Using pc1.csv and pc2, fit a surface to the data using the standard least square method and the total least square method. Plot the results (the surface) for each method and explain your interpretation of the results.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Coefficients of Standard Least Square plane equation for PC1 are:  [-0.35395482] [-0.66855145] [3.20255363]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Coefficients of Standard Least Square plane equation for PC2 are:  [-0.25188404] [-0.67173669] [3.66025669]

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221436499-28530693-74ef-4adf-9a97-f04bb51cd9aa.png" alt= “Standard_Least_Square” width="600" height="350">
</p>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Coefficients of Total Least Square plane equation for PC1 are:  0.28616427612095185 0.5397123383073391 0.7917200256094297 -2.5344641945425828

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Coefficients of Total Least Square plane equation for PC2 are:  -0.221074092839804 -0.5873941957270391 -0.7785205869476045 2.8494445218366775

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221436517-80152ecf-25ec-4ac8-b428-e6f7e3d2bd09.png" alt= “Total_Least_Square” width="600" height="350">
</p>

&nbsp;&nbsp;&nbsp;&nbsp; b. Additionally, fit a surface to the data using RANSAC. You will need to write RANSAC code from scratch. Briefly explain all the steps of your solution, and the parameters used. Plot the output surface on the same graph as the data. Discuss which graph fitting method would be a better choice of outlier rejection.

<p align="center">
<img src="https://user-images.githubusercontent.com/40595475/221436475-a250679e-d2bd-479e-b3d6-96d4433c33c8.png" alt= “RANSAC” width="600" height="350">
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RANSAC Surface equation coefficients for PC1 are: a = -82.80817673028923  b = -151.58171705887023 c = -208.86214701779943 d = 642.53322429545 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; RANSAC Surface equation coefficients for PC2 are: a = 57.73853977057806  b = 111.72937258701376 c = 169.93983739816065 d = -501.68562992996283 
