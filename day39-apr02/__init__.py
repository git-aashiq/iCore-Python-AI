import cv2
import numpy as np

# Load the image
image = cv2.imread('img1.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a binary image
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out contours based on length
min_length = 100  # Minimum length of a contour to be considered a dashed line
dashed_lines = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > min_length]

# Draw filtered dashed lines on the original image
cv2.drawContours(image, dashed_lines, -1, (0, 255, 0), 2)

# Count the number of dashed lines
num_dashed_lines = len(dashed_lines)
print("Number of dashed lines:", num_dashed_lines)

# Display the result
cv2.imshow('Detected Dashed Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()









#
# import cv2
# import numpy as np
#
# # Load the image
# image = cv2.imread('img1.jpg')
#
# # Convert image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply adaptive thresholding to create a binary image
# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#
# # Calculate horizontal projection profile
# horizontal_projection = np.sum(binary, axis=1)
#
# # Set a threshold for line detection based on projection profile
# line_threshold = np.max(horizontal_projection) * 0.9
#
# # Detect lines with high density
# line_indices = np.where(horizontal_projection > line_threshold)[0]
#
# # Draw detected lines on the original image
# for line in line_indices:
#     cv2.line(image, (0, line), (image.shape[1], line), (0, 255, 0), 2)
#
# # Display the result
# cv2.imshow('Detected Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
