import cv2
import numpy as np

def edge_mask(img,line_size,blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

path=input("Please enter the path: ")
img = cv2.imread(path)
blur_value = line_size = 7
edges = edge_mask(img, line_size, blur_value)
total_color = 8
k=total_color
data = np.float32(img).reshape((-1,3))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
ret, label , center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
result = center[label.flatten()]
result = result.reshape(img.shape)
real = input("Please enter the name you want to sace the file as: ")
real=real+".jpg"
cv2.imwrite(real, result)
