import cv2
import numpy as np
import os


def transform(pos):
    # This function is used to find the corners of the object and the dimensions of the object
    pts = []
    n = len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    sums = {}
    diffs = {}
    tl = tr = bl = br = 0
    for i in pts:
        x = i[0]
        y = i[1]
        sum_val = x + y
        diff = y - x
        sums[sum_val] = i
        diffs[diff] = i

    if not sums or not diffs:
        print("Error: No valid sums or diffs found.")
        return None, None, None

    sums = sorted(sums.items())
    diffs = sorted(diffs.items())
    n = len(sums)

    if n == 0:
        print("Error: Empty lists after sorting.")
        return None, None, None

    rect = [sums[0][1], diffs[0][1], diffs[n - 1][1], sums[n - 1][1]]
    # top-left top-right bottom-left bottom-right

    h1 = np.sqrt((rect[0][0] - rect[2][0]) * 2 + (rect[0][1] - rect[2][1]) * 2)  # height of left side
    h2 = np.sqrt((rect[1][0] - rect[3][0]) * 2 + (rect[1][1] - rect[3][1]) * 2)  # height of right side
    h = max(h1, h2)

    w1 = np.sqrt((rect[0][0] - rect[1][0]) * 2 + (rect[0][1] - rect[1][1]) * 2)  # width of upper side
    w2 = np.sqrt((rect[2][0] - rect[3][0]) * 2 + (rect[2][1] - rect[3][1]) * 2)  # width of lower side
    w = max(w1, w2)

    return int(w), int(h), rect


# Function to count number of lines or nozzles in the image
def count_lines(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    # Count the number of detected lines
    num_lines = 0
    if lines is not None:
        num_lines = len(lines)

    return num_lines


# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the file path using the absolute path
img_path = os.path.join(script_dir, 't5.jpg')

# Load the image
img = cv2.imread(img_path)

# Check if the image is loaded successfully
if img is None:
    print(f"Error: Unable to load the image at {img_path}")
else:
    # Resize the image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('INPUT', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edge = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, [0, 0, 255], 2)
    cv2.imshow('Contours', img)

    # Find the largest contour
    max_area = 0
    pos = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            pos = contour

    if pos is not None:
        peri = cv2.arcLength(pos, True)
        approx = cv2.approxPolyDP(pos, 0.02 * peri, True)

        size = img.shape
        w, h, arr = transform(approx)

        if w is not None and h is not None and arr is not None:
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts1 = np.float32(arr)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (w, h))
            cv2.imshow('OUTPUT', dst)
            cv2.imwrite("output.jpg", dst)

            # Count the number of lines in the image
            num_lines = count_lines(dst)
            print("Number of lines/nozzles in the image:", num_lines)
        else:
            print("Error: Failed to transform the image.")
    else:
        print("No contour found.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



