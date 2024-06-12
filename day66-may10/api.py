from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

# Global variables to store the coordinates
refPt = []
cropping = False
image = None

def click_and_crop(event, x, y, flags, param):
    # Grab references to the global variables
    global refPt, cropping, image

    # If the left mouse button was clicked, record the starting (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # If the mouse is being moved and cropping is in progress
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        # Draw a rectangle on the image
        image_with_rectangle = image.copy()
        cv2.rectangle(image_with_rectangle, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", image_with_rectangle)

    # If the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record the ending (x, y) coordinates
        refPt.append((x, y))
        cropping = False

        # Draw a final rectangle on the image
        image_with_rectangle = image.copy()
        cv2.rectangle(image_with_rectangle, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image_with_rectangle)


@app.route('/', methods=['GET'])
def home():
    return "Welcome to our image processing API!"

@app.route('/process', methods=['POST'])
def process_image():
    # Check if an image was posted
    if 'image' not in request.files:
        return "No image provided!", 400

    # Load the image
    global image
    image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Your image processing code here
    # For example, convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the processed image to a byte array
    is_success, buffer = cv2.imencode(".jpg", gray)
    io_buf = io.BytesIO(buffer)

    # Return the byte array as a file
    return send_file(io_buf, mimetype='image/jpeg')




@app.route('/crop', methods=['POST'])
def crop_image():
    global image
    # Check if an image was posted
    if 'image' not in request.files:
        return "No image provided!", 400

    # Load the image
    image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Create a window and set a mouse callback function
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)

    # Display the image
    cv2.imshow("image", image)


    # Wait for a keypress
    cv2.waitKey(0)

    # If the 'c' key is pressed, close window and return the cropped image
    if cv2.waitKey(0) & 0xFF == ord("c"):
        # Close the window
        cv2.destroyAllWindows()

        # Save the cropped image
        if len(refPt) == 2:
            roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            is_success, buffer = cv2.imencode(".jpg", roi)
            io_buf = io.BytesIO(buffer)
            return send_file(io_buf, mimetype='image/jpeg')
        else:
            return "Please select two points for cropping.", 400

if __name__ == "__main__":
    app.run(debug=True)
