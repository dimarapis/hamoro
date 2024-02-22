
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import os
import glob
import shutil

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
    
# Create a mouse callback function
def on_mouse_click(event, x, y, flags, param):
    global clicked_x, clicked_y  # Declare that these variables are from the outer scope
    global array
    global input_label
    global grayscale_save
    global combined_image
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_x = x
        clicked_y = y
        array = np.append(array,(x,y))
        input_label = np.append(input_label,1)
        num_rows = int(np.ceil(len(array) / 2))
        # Reshape to a 2D array with 2 columns and an appropriate number of rows
        reshaped_array = array.reshape(num_rows, 2)
        #input_label = np.array(len(reshaped_array) * [1])
        masks, scores, logits = predictor.predict(
            point_coords=reshaped_array,
            point_labels=input_label,
            multimask_output=True,
        )
        
        grayscale_image = np.uint8(masks[0]) * 255
        grayscale_save = grayscale_image.copy()
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
        grayscale_image[np.where((grayscale_image == [255,255,255]).all(axis = 2))] = [0,0,255]
        combined_image = cv2.addWeighted(orig_image, 1, grayscale_image, 0.7, 0)
        

        cv2.imshow('Image', combined_image)  

    elif event == cv2.EVENT_MBUTTONDOWN:
        clicked_x = x
        clicked_y = y
        array = np.append(array,(x,y))
        input_label = np.append(input_label,0)
        num_rows = int(np.ceil(len(array) / 2))
        # Reshape to a 2D array with 2 columns and an appropriate number of rows
        reshaped_array = array.reshape(num_rows, 2)
        #input_label = np.array(len(reshaped_array) * [1])
        masks, scores, logits = predictor.predict(
            point_coords=reshaped_array,
            point_labels=input_label,
            multimask_output=True,
        )
        
        grayscale_image = np.uint8(masks[0]) * 255
        grayscale_save = grayscale_image.copy()
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
        grayscale_image[np.where((grayscale_image == [255,255,255]).all(axis = 2))] = [0,0,255]
        combined_image = cv2.addWeighted(orig_image, 1, grayscale_image, 0.7, 0)
        

        cv2.imshow('Image', combined_image)  
    
    

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

for IMAGE_PATH in sorted(glob.glob('data/floor_input/*.png')):
    clicked_x = -1
    clicked_y = -1
    array = np.array([])
    input_label = np.array([])
    # load image
    #print(IMAGE_PATH)
    save_path_mask = IMAGE_PATH.replace('floor_input','floor_output')
    save_path_combined = IMAGE_PATH.replace('floor_input','floor_combined')
    
    image = cv2.imread(IMAGE_PATH)
    # Create a window and set the mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', on_mouse_click)

    # Display the image
    cv2.imshow('Image', image)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # Wait for a key press with a timeout
    key = cv2.waitKey(0)

    if key == ord('s'):  # Check if the 'q' key was pressed
        print('saving image')
        #move image to new folder
        shutil.move(IMAGE_PATH, IMAGE_PATH.replace('floor_input','floor_done'))
        cv2.imwrite(f'{save_path_mask}', grayscale_save)
        cv2.imwrite(f'{save_path_combined}', combined_image)
        #print(clicked_x,clicked_y)
        #print(array)
        # Determine the number of rows based on the length of the array
    if key == ord('d'):
        shutil.move(IMAGE_PATH, IMAGE_PATH.replace('floor_input','floor_missing'))

        #move image to new folder
        
        #shutil.copy()


    elif key == ord('q'):  # Check if the Escape key was pressed
            print("The Escape key was pressed.")
            break