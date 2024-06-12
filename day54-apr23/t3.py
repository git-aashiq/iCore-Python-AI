# import cv2
# import numpy as np
#
# # Read the input and template images
# input_image = cv2.imread('IMG_20240226_172115.jpg')
# template = cv2.imread('img.png')
#
# # Convert images to grayscale
# input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# 
# # Perform template matching
# result = cv2.matchTemplate(input_gray, template_gray, cv2.TM_CCOEFF_NORMED)
#
# # Get the coordinates of the best match
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# top_left = max_loc
# h, w = template_gray.shape
#
# # Draw a rectangle around the matched region
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(input_image, top_left, bottom_right, (0, 255, 0), 2)
#
# # Display the result
# cv2.imshow('Matched Image', input_image)
# cv2.resizeWindow('Matched Image', 600,600)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
