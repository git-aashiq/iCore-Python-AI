import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\User-PC\Downloads\IMG_20240226_172115.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to read the image from the path: {image_path}")
    exit()

# Resize the image to fit within the screen for easier manipulation
max_height = 1000
scale = max_height / image.shape[0]
image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
orig_image = image.copy()

# Define the initial points
points = np.array([[50, 50], [image.shape[1] - 50, 50], [image.shape[1] - 50, image.shape[0] - 50], [50, image.shape[0] - 50]], dtype='float32')

# Variables to store the state
selected_point = None
radius = 10  # Radius of corner points for selection
color = (0, 255, 0)  # Color of the corner points and lines

def draw_points(image, points):
    """Draw points and connecting lines on the image."""
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    cv2.polylines(image, [points.astype(np.int32)], isClosed=True, color=color, thickness=2)

def mouse_callback(event, x, y, flags, param):
    global selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(points):
            if np.linalg.norm(point - [x, y]) < radius:
                selected_point = i
                break
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = None
    elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
        points[selected_point] = [x, y]

# Create a window and set the mouse callback function
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image', mouse_callback)

while True:
    # Clone the original image to draw on
    temp_image = orig_image.copy()

    # Draw points and connecting lines
    draw_points(temp_image, points)

    # Display the image
    cv2.imshow('Image', temp_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Crop the region inside the four points
points = points.astype(np.int32)
rect = cv2.boundingRect(points)
x, y, w, h = rect
cropped_image = orig_image[y:y+h, x:x+w]

# Save and display the cropped image
cropped_image_path = 'cropped_image.png'
cv2.imwrite(cropped_image_path, cropped_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Cropped image saved to: {cropped_image_path}")
