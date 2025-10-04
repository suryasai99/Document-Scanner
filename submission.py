# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt 


# reading image
img = cv2.imread('/Users/suryasaikadali/Downloads/open_Cv/CVIP/project_3/scanned-form.jpg')

# converting to grey scale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, thresh = cv2.threshold(img_grey, 200,255, cv2.THRESH_BINARY)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# select the contour with maximum area
max_contour = max(contours, key = cv2.contourArea)

# finding the four corner points of the contour
peri = cv2.arcLength(max_contour, True)
approx = cv2.approxPolyDP(max_contour, 0.02*peri, True)
for i in approx:
    approx_1 = np.array([i[0].tolist() for i in approx], dtype="float32")

# function to order points in clockwise order
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

new_pts = order_points(approx_1)

# calculating aspect ratio of the image
aspect_ratio = img.shape[0]/img.shape[1]

# setting desired width and height
desired_width = 500
desired_height = int(desired_width * aspect_ratio)

pts = np.array([[0,0],
               [desired_width-1, 0],
               [desired_width-1, desired_height-1],
               [0, desired_height-1]], dtype="float32")

# finding perspective transform matrix
pers_trans= cv2.getPerspectiveTransform(new_pts,pts)

# applying perspective transformation
res_1 = cv2.warpPerspective(img, pers_trans, (desired_width, desired_height))

# saving the output
cv2.imwrite('/Users/suryasaikadali/Downloads/open_Cv/CVIP/project_3/scanned_output.jpg', res_1)

# displaying the output
cv2.imshow('output', res_1)
cv2.waitKey(0)
cv2.destroyAllWindows()