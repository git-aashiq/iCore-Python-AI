# import numpy as np
# import cv2
# from PIL import Image
#
# file_name = "pict3.jpg"
#
# def opencv_resize(image, ratio):
#     width = int(image.shape[1] * ratio)
#     height = int(image.shape[0] * ratio)
#     dim = (width, height)
#     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
# image = cv2.imread(file_name)
# resize_ratio = 500 / image.shape[0]
# original = image.copy()
# image = opencv_resize(image, resize_ratio)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# dilated = cv2.dilate(blurred, rectKernel)
# edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
#
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#
# def approximate_contour(contour):
#     peri = cv2.arcLength(contour, True)
#     return cv2.approxPolyDP(contour, 0.032 * peri, True)
#
# def get_receipt_contour(contours):
#     for c in contours:
#         approx = approximate_contour(c)
#         if len(approx) == 4:
#             return approx
#
# receipt_contour = get_receipt_contour(largest_contours)
#
# def contour_to_rect(contour):
#     pts = contour.reshape(4, 2)
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     return rect / resize_ratio
#
# def wrap_perspective(img, rect):
#     (tl, tr, br, bl) = rect
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA) if not np.isnan(heightA) else 0, int(heightB) if not np.isnan(heightB) else 0)
#     maxWidth = max(int(widthA), int(widthB))
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#     M = cv2.getPerspectiveTransform(rect, dst)
#     return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
#
# scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour))
#
# def bw_scanner(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return result
#
# result = bw_scanner(scanned)
#
# output = Image.fromarray(result)
#
# output.show()
# output.save('result.jpg')













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

    h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2)  # height of left side
    h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 + (rect[1][1] - rect[3][1]) ** 2)  # height of right side
    h = max(h1, h2)

    w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2)  # width of upper side
    w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 + (rect[2][1] - rect[3][1]) ** 2)  # width of lower side
    w = max(w1, w2)

    return int(w), int(h), rect


# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the file path using the absolute path
img_path = os.path.join(script_dir, 't3.jpg')

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
            image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imshow('OUTPUT', image)
            cv2.imwrite("output.jpg", image)
        else:
            print("Error: Failed to transform the image.")
    else:
        print("No contour found.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()





























# import cv2
# import numpy as np
# import os
#
# def transform(pos, img_shape):
#     # This function is used to find the corners of the object and the dimensions of the object
#     pts = []
#     n = len(pos)
#     for i in range(n):
#         pts.append(list(pos[i][0]))
#
#     sums = {}
#     diffs = {}
#     tl = tr = bl = br = 0
#     for i in pts:
#         x = i[0]
#         y = i[1]
#         sum_val = x + y
#         diff = y - x
#         sums[sum_val] = i
#         diffs[diff] = i
#     sums = sorted(sums.items())
#     diffs = sorted(diffs.items())
#     n = len(sums)
#     rect = [sums[0][1], diffs[0][1], diffs[n-1][1], sums[n-1][1]]
#     # top-left top-right bottom-left bottom-right
#
#     h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2)  # height of left side
#     h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 + (rect[1][1] - rect[3][1]) ** 2)  # height of right side
#     h = max(h1, h2)
#
#     w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2)  # width of upper side
#     w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 + (rect[2][1] - rect[3][1]) ** 2)  # width of lower side
#     w = max(w1, w2)
#
#     return int(w), int(h), rect
#
# # Get the absolute path to the script's directory
# script_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Specify the file path using the absolute path
# img_path = os.path.join(script_dir, 'pict2.jpg')
#
# # Load the image
# img = cv2.imread(img_path)
#
# # Check if the image is loaded successfully
# if img is None:
#     print(f"Error: Unable to load the image at {img_path}")
# else:
#     # Resize the image
#     r = 500.0 / img.shape[1]
#     dim = (500, int(img.shape[0] * r))
#     img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
#     cv2.imshow('INPUT', img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (11, 11), 0)
#     edge = cv2.Canny(gray, 100, 200)
#
#     # Find contours
#     contours, _ = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img, contours, -1, [0, 0, 255], 2)
#     cv2.imshow('Contours', img)
#
#     # Find the largest contour
#     max_area = 0
#     pos = 0
#     for i in contours:
#         area = cv2.contourArea(i)
#         if area > max_area:
#             max_area = area
#             pos = i
#
#     peri = cv2.arcLength(pos, True)
#     approx = cv2.approxPolyDP(pos, 0.02 * peri, True)
#
#     size = img.shape
#     w, h, arr = transform(approx, size)
#
#     # Calculate the position of the object relative to the center of the image
#     center_x = size[1] // 2
#     object_center_x = sum([p[0] for p in arr]) // 4
#     if object_center_x < center_x:
#         # Object is towards the left side
#         # Adjust the perspective transformation accordingly
#         arr[0], arr[1] = arr[1], arr[0]
#         arr[2], arr[3] = arr[3], arr[2]
#     else:
#         # Object is towards the right side
#         # Adjust the perspective transformation accordingly
#         arr[1], arr[2] = arr[2], arr[1]
#         arr[0], arr[3] = arr[3], arr[0]
#
#     pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#     pts1 = np.float32(arr)
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     dst = cv2.warpPerspective(img, M, (w, h))
#     image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#     if w > 0 and h > 0:
#         image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
#     else:
#         print("Invalid dimensions provided for resizing.")
#     cv2.imshow('OUTPUT', image)
#     cv2.imwrite("output.jpg", image)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
