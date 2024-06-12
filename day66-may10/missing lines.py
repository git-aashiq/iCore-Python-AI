import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained CNN model for line detection
model = tf.keras.models.load_model('line_detection_model.h5')

# Load image
image = cv2.imread('scr.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Edge detection (Canny edge detection)
edges = cv2.Canny(image, 50, 150)

# Step 2: Probabilistic Hough transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Step 3: Deep learning-based line detection
# Preprocess image for CNN input
resized_image = cv2.resize(image, (224, 224))
normalized_image = resized_image / 255.0
input_image = np.expand_dims(normalized_image, axis=0)

# Predict lines using pre-trained CNN model
predicted_lines = model.predict(input_image)

# Threshold the predicted lines to get binary mask
threshold = 0.5
binary_mask = (predicted_lines > threshold).astype(np.uint8) * 255

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Edge Detection', edges)
cv2.imshow('Probabilistic Hough Transform', np.zeros_like(edges))  # Display Hough lines on edge image
cv2.imshow('Deep Learning Line Detection', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
