# Inspired by and based on: https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
# To download the caffemodel and the prototxt, see: https://github.com/richzhang/colorization/tree/caffe/colorization/models
# To download pts_in_hull.npy, see: https://github.com/richzhang/colorization/tree/caffe/colorization/resources/pts_in_hull.npy

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import random

# Function to center the window on the screen
def center_window_on_screen():

    # Calculate the x and y coordinates for centering
    x_cord = int((screen_width / 2) - (width / 2))
    y_cord = int((screen_height / 2) - (height / 2))

    # Set the geometry of the root window to center it
    root.geometry("{}x{}+{}+{}".format(width, height, x_cord, y_cord))


# Function to select an image file using file dialog
def select_img():
    global img

    # Open a file dialog to choose an image file
    filename = filedialog.askopenfilename(
        initialdir='/images',
        title="Select Image",
        filetypes=(("all files", "*.*"), ("png images", "*.png"), ("jpg images", "*.jpg"))
    )

    # Open and resize the selected image using PIL
    img = Image.open(filename)
    img = img.resize((400, 400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    # Display the selected image in the tkinter Label
    lbl_show_img['image'] = img

    # Insert the image path into the entry widget
    entry_img_path.delete(0, tk.END) # clear the entry field
    entry_img_path.insert(0, filename)


# Function to colorize the selected image
def colorize_img():
    global img

    # Change the background color of the buttons to a random color when clicked
    random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    btn_colorize.config(bg=random_color)
    btn_select_img.config(bg=random_color)
    btn_exit.config(bg=random_color)

    # Paths to the model and the selected image from the entry widget
    prototxt_path = 'model/colorization_deploy_v2.prototxt'
    model_path = 'model/colorization_release_v2.caffemodel'
    kernel_path = 'model/pts_in_hull.npy'
    image_path = entry_img_path.get()

    # Select desired model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path) # load cluster centers

    # Populate cluster centers as 1x1 convolution kernel
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Read the selected image and preprocess it
    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32") / 255.0 # convert pixel values to folating point format and normalize to the range [0,1]
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB) # convert the image from RGB to Lab
    resized = cv2.resize(lab, (224, 224)) # resizing the image to standard size for the pretrained model
    L = cv2.split(resized)[0] # pull out L channel
    L -= 50 # subtract 50 for mean-centering

    # Set the input for the colorization model
    net.setInput(cv2.dnn.blobFromImage(L))

    # Predict the ab channels
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0)) # this is our result

    # Resize the predicted ab channels to match the original image size
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    # Combine L channel and predicted ab channels to get colorized image
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    # Save the colorized image
    cv2.imwrite('colorized_images/colorized_image.jpg', colorized)

    # Load the saved image using PIL and create a Tkinter PhotoImage
    saved_image = Image.open('colorized_images/colorized_image.jpg')
    saved_image = saved_image.resize((400, 400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(saved_image)

    # Display the colorized image in the tkinter Label
    lbl_show_img['image'] = img


# Create a tkinter root window
root=tk.Tk()

# Define width and height of the window
width, height = 600, 650

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Center the window on the screen
center_window_on_screen()

# Set the title and geometry of the root window
root.title("Image colorization")
root.geometry("600x600")

# Create a frame to contain widgets
frame=tk.Frame(root)

# Create labels, entry widget, and buttons
#img path
lbl_img_path=tk.Label(frame, text='Image path: ', font=('Sans Serif', 16), padx=25, pady=25)
entry_img_path=tk.Entry(frame, font=('Sans Serif', 11), width=40)
#show img
lbl_show_img=tk.Label(frame)
#buttons
btn_select_img=tk.Button(frame, text='Select Image', bg='grey', fg='#ffffff',
                         font=('Sans Serif', 12), padx=10, command=select_img)
btn_colorize=tk.Button(frame, text='Colorize', bg='grey', fg='#ffffff',
                       font=('Sans Serif', 12), padx=27, command=colorize_img)
btn_exit=tk.Button(frame, text='Exit', bg='grey', fg='#ffffff',
                   font=('Sans Serif', 12), padx=40, command=lambda:exit())

# Pack widgets into the frame
frame.pack()
lbl_img_path.grid(row=0, column=0)
entry_img_path.grid(row=0, column=1, padx=(0, 10), columnspan=64)
lbl_show_img.grid(row=1, column=0, columnspan=4)
btn_select_img.grid(row=2, column=0, padx=10,pady=35, ipadx=13)
btn_colorize.grid(row=2, column=1, padx=10, pady=35, ipadx=13)
btn_exit.grid(row=2, column=2, padx=10, pady=35, ipadx=13)

# Start the tkinter event loop
root.mainloop()