import cv2
import numpy as np


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


# Set up configuration
imagePaths = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']   # Path to the image file

# Initialize image processing parameters
heightImg = 1080  # Desired height of the image
widthImg = 1920  # Desired width of the image
blur_kernel_size = (9, 9)  # Kernel size for Gaussian blur
canny_thresholds = (10, 150)  # Thresholds for Canny edge detection
dilation_kernel = np.ones((5, 5), np.uint8)  # Kernel for dilation
erosion_kernel = np.ones((5, 5), np.uint8)  # Kernel for erosion


for i, pathImage in enumerate(imagePaths):
    # Read the input image
    img = cv2.imread(pathImage)  # Read image from the specified file
    img = cv2.resize(img, (widthImg, heightImg))  # Resize the input image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, blur_kernel_size, 0)  # Apply Gaussian blur to the grayscale image
    imgThreshold = cv2.Canny(imgBlur, canny_thresholds[0], canny_thresholds[1])  # Apply Canny edge detection
    imgDial = cv2.dilate(imgThreshold, dilation_kernel, iterations=2)  # Apply dilation
    imgThreshold = cv2.erode(imgDial, erosion_kernel, iterations=1)  # Apply erosion

    # Find all contours
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find all contours

    # Find the biggest contour
    biggest, maxArea = biggestContour(contours)  # Find the biggest contour
    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)  # Prepare points for warp
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Prepare points for warp
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Calculate the perspective transform matrix
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # Apply perspective transform

        # Remove 20 pixels from each side
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 5, 5:imgWarpColored.shape[1] - 5]
        imgWarpColored = cv2.resize(imgWarpColored, (400, 600))  # Resize the warped image to a smaller size

        # Display the final processed image
        cv2.imshow("result" + str(i+1) + ".jpg", imgWarpColored)
        cv2.imwrite("result" + str(i + 1) + ".jpg", imgWarpColored)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cv2.destroyAllWindows()
