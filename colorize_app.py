import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import argparse
import torch
from torchvision.utils import save_image
from network import ColorizeNet
from utils import load_gray, to_rgb

class ColorizeApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Colorization")

        self.selected_img_label = tk.Label(master)
        self.selected_img_label.pack()

        self.colorized_img_label = tk.Label(master)
        self.colorized_img_label.pack()

        self.select_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.colorize_button = tk.Button(master, text="Colorize", command=self.colorize)
        self.colorize_button.pack()

        self.model = ColorizeNet()
        self.model.load_state_dict(torch.load('models/model.pth', map_location='cpu'))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.selected_img = Image.open(file_path)
            self.selected_img.thumbnail((400, 400)) 
            self.selected_img_tk = ImageTk.PhotoImage(self.selected_img)
            self.selected_img_label.config(image=self.selected_img_tk)
            self.selected_img_path = file_path
            self.colorize_button.config(state=tk.NORMAL)
    def colorize(self):
        if hasattr(self, 'selected_img_path'):
            img_l = load_gray(self.selected_img_path, shape=360)

            self.model.eval()
            with torch.no_grad():
                img_ab = self.model(img_l)

            img_rgb = to_rgb(img_l, img_ab)
            img_rgb = (img_rgb * 255).astype('uint8') 
            colorized_img = Image.fromarray(img_rgb)

            colorized_img.thumbnail((400, 400)) 
            self.colorized_img_tk = ImageTk.PhotoImage(colorized_img)
            self.colorized_img_label.config(image=self.colorized_img_tk)
            self.colorize_button.config(state=tk.DISABLED)
        else:
            print("Please select an image first.")
def main():
    root = tk.Tk()
    app = ColorizeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
