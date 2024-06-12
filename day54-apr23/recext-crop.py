import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pyimagesearch import transform
from pyimagesearch import imutils

class ManualCropApp:
    def __init__(self, master):
        self.master = master
        master.title("Manual Crop")

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.canvas = tk.Canvas(master)
        self.canvas.pack()

        self.image = None
        self.original_image = None
        self.start_point = None
        self.rect_id = None

        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)

        self.save_button = tk.Button(master, text="Save", command=self.save_cropped_image)
        self.save_button.pack()

    def save_cropped_image(self, cropped_image):
        if cropped_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, cropped_image)


    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image.size > 1024*1024:  # Check if image size is greater than 1MB
                self.original_image = self.resize_image(self.original_image, width=600, height=600)

            # Apply perspective transformation
            self.image = self.apply_perspective_transform(self.original_image)

            self.display_image()

    def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def apply_perspective_transform(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Iterate over the contours and find the largest one that approximates a quadrilateral
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx.reshape(4, 2)
                warped = transform.four_point_transform(image, screenCnt)
                return warped

        # If no suitable contour found, return original image
        return image

    def display_image(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(image_pil)
        self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

    def start_rect(self, event):
        self.start_point = (event.x, event.y)

    def draw_rect(self, event):
        if self.start_point:
            x0, y0 = self.start_point
            x1, y1 = event.x, event.y
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red")

    def end_rect(self, event):
        if self.start_point:
            x0, y0 = self.start_point
            x1, y1 = event.x, event.y
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red")

            # Crop the image
            x_min = min(x0, x1)
            x_max = max(x0, x1)
            y_min = min(y0, y1)
            y_max = max(y0, y1)
            cropped_image = self.image[y_min:y_max, x_min:x_max]

            # Display the cropped image
            self.display_cropped_image(cropped_image)
            self.save_cropped_image(cropped_image)





    def display_cropped_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(image_pil)
        self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)


root = tk.Tk()
app = ManualCropApp(root)
root.mainloop()
