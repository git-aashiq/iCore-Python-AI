import cv2
import numpy as np

# Load the input image
image_path = r"C:\Users\User-PC\Downloads\IMG_20240226_173249\IMG_20240226_172115.jpg"
image = cv2.imread(image_path)

# Resize the image
orig = image.copy()
(H, W) = image.shape[:2]
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# Resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Define the output layer names for the EAST detector model
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# Load the pre-trained EAST text detector
net = cv2.dnn.readNet(r"C:\Users\User-PC\Downloads\frozen_east_text_detection.pb")

# Construct a blob from the image and then perform a forward pass of the model
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# Decode the predictions
(numRows, numCols) = scores.shape[2:4]
rects = []

# Loop over the number of rows
for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        if scoresData[x] < 0.1:
            continue

        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        rects.append((startX, startY, endX, endY))

# Function to detect lines using Hough Transform
def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    return lines

# Detect lines in the original image
lines = detect_lines(orig)

# Convert lines to bounding boxes
line_boxes = []
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_boxes.append((x1, y1, x2, y2))

# Get the bounding box for the receipt
if len(rects) > 0 or len(line_boxes) > 0:
    points = []
    for rect in rects:
        startX, startY, endX, endY = rect
        points.append([startX, startY])
        points.append([endX, endY])
    for box in line_boxes:
        x1, y1, x2, y2 = box
        points.append([x1, y1])
        points.append([x2, y2])
    points = np.array(points, dtype='int')

    # Apply minimum area rectangle to get the outer bounds of the detected points
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Order the points in the box
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]

    # Get the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Set up the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Apply the perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # Save the cropped image
    cv2.imwrite('cropped_receipt.jpg', warped)
else:
    print("No text or lines were detected in the image.")
