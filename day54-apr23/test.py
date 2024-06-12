# import cv2
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
#
# class ManualCropApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("Manual Crop")
#
#         self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
#         self.load_button.pack()
#
#         self.canvas = tk.Canvas(master)
#         self.canvas.pack()
#
#         self.image = None
#         self.start_point = None
#         self.rect_id = None
#
#         self.canvas.bind("<Button-1>", self.start_rect)
#         self.canvas.bind("<B1-Motion>", self.draw_rect)
#         self.canvas.bind("<ButtonRelease-1>", self.end_rect)
#
#     def load_image(self):
#         file_path = filedialog.askopenfilename()
#         if file_path:
#             self.image = cv2.imread(file_path)
#             if self.image.size > 1024*1024:  # Check if image size is greater than 1MB
#                 self.image = self.resize_image(self.image, width=600, height=600)
#             self.display_image()
#
#     def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
#         dim = None
#         (h, w) = image.shape[:2]
#
#         if width is None and height is None:
#             return image
#
#         if width is None:
#             r = height / float(h)
#             dim = (int(w * r), height)
#
#         else:
#             r = width / float(w)
#             dim = (width, int(h * r))
#
#         resized = cv2.resize(image, dim, interpolation=inter)
#         return resized
#
#     def display_image(self):
#         image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
#         image_pil = Image.fromarray(image_rgb)
#         self.image_tk = ImageTk.PhotoImage(image_pil)
#         self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
#
#     def start_rect(self, event):
#         self.start_point = (event.x, event.y)
#
#     def draw_rect(self, event):
#         if self.start_point:
#             x0, y0 = self.start_point
#             x1, y1 = event.x, event.y
#             if self.rect_id:
#                 self.canvas.delete(self.rect_id)
#             self.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red")
#
#     def end_rect(self, event):
#         if self.start_point:
#             x0, y0 = self.start_point
#             x1, y1 = event.x, event.y
#             if self.rect_id:
#                 self.canvas.delete(self.rect_id)
#             self.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red")
#
#             # Crop the image
#             x_min = min(x0, x1)
#             x_max = max(x0, x1)
#             y_min = min(y0, y1)
#             y_max = max(y0, y1)
#             cropped_image = self.image[y_min:y_max, x_min:x_max]
#             self.display_cropped_image(cropped_image)
#
#     def display_cropped_image(self, image):
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_pil = Image.fromarray(image_rgb)
#         self.image_tk = ImageTk.PhotoImage(image_pil)
#         self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
#
#
# root = tk.Tk()
# app = ManualCropApp(root)
# root.mainloop()