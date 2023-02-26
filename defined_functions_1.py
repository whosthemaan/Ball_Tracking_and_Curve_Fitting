import numpy as np
import matplotlib.pyplot as plt
import cv2

#loads all the images in vid into a list
def load_images(vid):
    i=0
    cv_img = []
    success, frame = vid.read()
    success = True
    while success:
        success, frame = vid.read()
        if success == True:
            cv_img.append(frame) 
    print("Successfully loaded %d images from the video" % len(cv_img))
    return cv_img

#Function to calculate coefficient of quadratic equation for the given set of x,y values
def least_square_fit_curve(x_data, y_data):
    X = np.ones((len(x_data), 3))
    Y = y_data

    #forming X matrix
    for j in range(2):
        X[:, j] = x_data ** (2 - j)

    # Solve the least-squares problem
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

    return coefficients

#This function plots the curves and makes use of coefficients and given x and y values
def plot_curve(x_data, y_data, coefficients):
    x_values = np.linspace(x_data.min(), x_data.max(), 100)

    y_values = np.zeros(x_values.size)

    for j in range(3):
        y_values += coefficients[j] * x_values** (2 - j)

    plt.plot(x_data, y_data, '.', label='Data points')
    plt.plot(x_values, y_values, 'r', label='Fitted curve')
    plt.legend()
    plt.show()

#takes average of the index of x,y for non zero values
def calculate_centroid(image):
  y_values, x_values = np.nonzero(image)
  X = np.sum(x_values)/x_values.shape[0]
  Y = np.sum(y_values)/y_values.shape[0]
  return (int(X), int(Y))

# filters red within specified range for each frame and calls centroid function
def get_centroids(video_frames):
    coordinates = []
    for images in video_frames[:]:
        mask = cv2.inRange(images, np.array([0, 0, 130]),  np.array([65, 65, 255]))
        output = cv2.bitwise_and(images, images, mask = mask) 
        gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,0,255,0)
        try:
            X, Y = calculate_centroid(thresh)
            cv2.circle(images, (X, Y), 5, (255, 255, 255), -1)
            coordinates.append((X,Y))
        except:
            continue
    return coordinates

