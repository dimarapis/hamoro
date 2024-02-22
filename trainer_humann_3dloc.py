import torch
from torch.utils.data import Dataset, DataLoader
from dataloader_eh import DepthDataset
import cv2 
import time
from my_cal_loco import Loco as Loco2
import wandb 
from humaNN_model import Loco
#from openpifpaf import decoder, network, visualizer, show, logger, Predictor
#from openpifpaf import datasets
#from openpifpaf.predict import out_name
import os
from ultralytics import YOLO
import numpy as np
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from matplotlib import pyplot as plt
import torch.nn as nn
import shutil
import sys
from PIL import Image

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import random
#Ignore warnings   
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")

from omegaconf import OmegaConf
import argparse
os.environ['CUDA_LAUNCH_BLOCKING']='1'

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/trainer.yaml')
    
opt = OmegaConf.load(parser.parse_args().config)

#model_selection = 'humann'#'monoloco' #'monoloco_pp','copied_monoloco','humann',monoloco_pretrained'

batch_size = opt.training.batch_size
learning_rate = opt.training.learning_rate
epochs = opt.training.epochs
show_table = opt.training.show_table
save_samples = opt.training.save_samples
num_stage = opt.training.num_stage

model_selection = opt.network.model_selection #'monoloco'#'monoloco' #'monoloco_pp','copied_monoloco','humann',monoloco_pretrained'
dataset_folder = opt.data.dataset_folder
test_dataset_folder = opt.data.test_dataset_folder


def extract_values(tensor_list):
    samples_data = []
    num_samples = len(tensor_list[0][0])
    #print(num_samples)
    for sample_idx in range(num_samples):
        sample_data = {
            'x': [],
            'y': [],
            'confidence': [],
            'depth': []
        }
        
        for keypoint_data in tensor_list:
            #print(keypoint_data)
            sample_data['x'].append(keypoint_data[0][sample_idx].item())
            sample_data['y'].append(keypoint_data[1][sample_idx].item())
            sample_data['confidence'].append(keypoint_data[2][sample_idx].item())
            sample_data['depth'].append(keypoint_data[3][sample_idx].item())
        
        samples_data.append(sample_data)
        #samples_data
    return samples_data
    
def multi_run_inference(image):
    pose_output = pose_model(image,)
    seg_output = seg_model(image)
    return pose_output, seg_output, image

def log_lr_changes(optimizer, last_lr, epoch, loss):
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != last_lr:
        #print(f'Learning rate changed from {last_lr} to {current_lr} at epoch {epoch}')
        lr_list.append((epoch, current_lr, loss))
    return current_lr
  
def yolov8_output(output, image):
    kpts_array = []
    with torch.no_grad():
        #print(f'output shape: {output.shape}')
        #print(output)
        output = torch.from_numpy(output).unsqueeze(0).to(device)
        #print(output)
        output = output_to_keypoint(output)  
        #print(output)
    for idx in range(output.shape[0]):
        nimg, kpts = plot_skeleton_kpts(image, output[idx, 7:].T, 3)
        kpts_array.append(kpts)       
    total_keypoints = kpts_array     
        
    return nimg, total_keypoints 

def l1_loss_from_laplace(pred, gt, print_loss = False):
    """Only for evaluation"""
    abs_diff = torch.abs(pred - gt)
    #print(abs_diff)
    #print(torch.mean(abs_diff))
    loss = torch.mean(torch.abs(pred - gt))
    if print_loss:
        print("*"*40+"L1 Loss"+"*"*40)
        print("*"*40+"L1 Loss"+"*"*40)
        print('pred',pred)
        print('gt',gt)
        print('abs_diff',abs_diff)
        print('mean',torch.mean(abs_diff))
        print("*"*40+"L1 Loss"+"*"*40)
        print("*"*40+"L1 Loss"+"*"*40)
    #print(pred)
    #print(gt)
    return loss

def weighted_l1_loss_from_laplace(pred, gt):
    """
    Compute a custom L1 loss between predictions and ground truth.
    
    Args:
    pred (torch.Tensor): The predicted values.
    gt (torch.Tensor): The ground truth values.
    
    Returns:
    torch.Tensor: The custom L1 loss.
    """
    eps = 1e-6  # small constant to prevent division by zero
    abs_diff = torch.abs(pred - gt)
    weights = 1.0 / (abs_diff + eps)  # calculate weights based on inverse of absolute differences
    weighted_loss = abs_diff * weights  # apply weights
    #print(abs_diff)
    #print(torch.mean(weighted_loss))
    return torch.mean(weighted_loss)  # return the mean of the weighted loss

def training_prediction(filepath, image, depth, keypoints, bbox,status):
    tensor_data = [torch.stack(item) for item in keypoints]
    tensor_data = torch.stack(tensor_data)
    tensor_data = tensor_data.permute(2,1,0)
    #print(tensor_data.shape)
    #print(tensor_data)
    print(tensor_data.shape)
    print(bbox)
    print(status)
    dic_out = locomodel.forward(tensor_data, K_original,bbox,status)
    #print(dic_out)
    def analysis(tensor_data):
        #get depth vlaues of tensor data
        depth_info = tensor_data[:,3,:]
        #count how many non zero values per tensor and put it in a list
        non_zero_tensors = []
        for tensor in depth_info:
            non_zero = torch.count_nonzero(tensor)
            # Step 1: Find indices of non-zero values
            non_zero_indices = torch.nonzero(tensor)
            # Step 2: Extract non-zero values using the indices
            non_zero_values = tensor[non_zero_indices]
            # Step 3: Calculate the mean of non-zero values
            mean_value = torch.mean(non_zero_values.float())
            non_zero_tensors.append((non_zero,mean_value))
            
        return non_zero_tensors
    dic_out['analysis'] = analysis(tensor_data)
    return dic_out



#locomodel = Loco2(mode='mono', net='monoloco', device=device, n_dropout = 0, p_dropout=0)
#locomodel = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net='monoloco', device=device, n_dropout = 0, p_dropout=0)
locomodel = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net=model_selection, device=device, n_dropout = 0, p_dropout=0, num_stage=num_stage)
#model_selection = 'monoloco_trained'
#locomodel = torch.load(f'{test_dataset_folder}/{model_selection}_best_model.pth')
#locomodel.load(f'{test_dataset_folder}/{model_selection}_best_model.pth')

dataset = DepthDataset(dataset_folder)
test_dataset = DepthDataset(test_dataset_folder)


K_original = [[599.978, 0.0, 318.604], [0.0, 600.5, 247.77], [0.0, 0.0, 1.0]]

#check if folder exists
if save_samples:
    if not os.path.exists(f'{dataset_folder}/{model_selection}_results'):
        #Copy folder
        shutil.copytree(f'{dataset_folder}/rgb', f'{dataset_folder}/{model_selection}_results')
    else:
        #Inform that folder already exists and asks if it should be overwritten
        #user_input = input(f'Folder {dataset_folder}/{model_selection}_results already exists. Do you want to overwrite it? (y/n): ')
        user_input = 'y'
        
        while user_input != 'y' and user_input != 'n':
            user_input = input(f'Type "y" or "n" for yes or no (y/n): ')
        if user_input == 'y':
            shutil.rmtree(f'{dataset_folder}/{model_selection}_results')
            shutil.copytree(f'{dataset_folder}/rgb', f'{dataset_folder}/{model_selection}_results')
        else:
            sys.exit('Stopping the program')
    if not os.path.exists(f'{test_dataset_folder}/{model_selection}_results'):
        #Copy folder
        shutil.copytree(f'{test_dataset_folder}/rgb', f'{test_dataset_folder}/{model_selection}_results')
    else:
        #Inform that folder already exists and asks if it should be overwritten
        #user_input = input(f'Folder {test_dataset_folder}/{model_selection}_results already exists. Do you want to overwrite it? (y/n): ')
        while user_input != 'y' and user_input != 'n':
            user_input = input(f'Type "y" or "n" for yes or no (y/n): ')
        if user_input == 'y':
            shutil.rmtree(f'{test_dataset_folder}/{model_selection}_results')
            shutil.copytree(f'{test_dataset_folder}/rgb', f'{test_dataset_folder}/{model_selection}_results')
        else:
            sys.exit('Stopping the program')
    if not os.path.exists(f'{test_dataset_folder}/{model_selection}_visual'):
        #Copy folder
        os.mkdir(f'{test_dataset_folder}/{model_selection}_visual')
        #shutil.copytree(f'{test_dataset_folder}/rgb', f'{test_dataset_folder}/{model_selection}_results')
    else:
        #Inform that folder already exists and asks if it should be overwritten
        #user_input = input(f'Folder {test_dataset_folder}/{model_selection}_visual already exists. Do you want to overwrite it? (y/n): ')
        while user_input != 'y' and user_input != 'n':
            user_input = input(f'Type "y" or "n" for yes or no (y/n): ')
        if user_input == 'y':
            shutil.rmtree(f'{test_dataset_folder}/{model_selection}_visual')
            os.mkdir(f'{test_dataset_folder}/{model_selection}_visual')
        else:
            sys.exit('Stopping the program')
    
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
visual_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if model_selection in ['identity', 'two_stage']:
    optimizer = torch.optim.Adam(list(locomodel.model.parameters()) + list(locomodel.auxmodel.parameters()), lr=learning_rate)
else:
    optimizer = torch.optim.Adam(locomodel.model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
last_lr = optimizer.param_groups[0]['lr']
lr_list = [(0, last_lr)]

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


def train_loop(dataloader,last_lr):    
    pred_values = []
    gt_depth_values = []
    epoch_loss = 0
    kpts_list = []
    incremental_file_list = 0
    for i, (file_path, images, depth_values, keypoints, bbox) in enumerate((dataloader)):
        #print(i,'filepath')
        #print(file_path,images,depth_values,keypoints,bbox)
        #print(file_path[0])
        #print(file_path)
        #print(bbox)
        gt_depth = depth_values[0]/100 
        #print(depth_values)
        #print()
        #print("NEW_BATCH")
        #print(f'gt_depth: {gt_depth.flatten()}')
        dic_out = training_prediction(file_path,images, depth_values, keypoints, bbox,status='train')
        #print()

        #print(dic_out)
        kpts_list.append(dic_out['analysis'])



        pred_depth = dic_out['d'].squeeze(1)
        optimizer.zero_grad()
        #print(pred_depth.shape, gt_depth.shape)
        #loss = weighted_l1_loss_from_laplace(pred_depth, gt_depth.to(device))  
        print(f'gt_depth: {gt_depth.flatten()}')

        loss = l1_loss_from_laplace(pred_depth, gt_depth.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()   
        # Append values to lists
        pred_values.extend(pred_depth.detach().cpu().flatten())
        gt_depth_values.extend(gt_depth.detach().cpu())
    
        if epoch % 10 == 0:
            pred_values_array = np.array(pred_values)
            pred_values_formatted = ["{:.2f}".format(value) for value in pred_values_array]
            gt_depth_values_array = np.array(gt_depth_values)
            error = pred_values_array - gt_depth_values_array
            error_formatted = ["{:.2f}".format(value) for value in error]
            epoch_loss_formatted = "{:.2f}".format(epoch_loss)
            if save_samples:
                for file_path_file in file_path:
                    formatted_file_path = file_path_file.split('/')[-1].replace('.txt', '_rgb.png')
                    image = cv2.imread(f'{dataset_folder}/rgb/{formatted_file_path}')
                    #Draw pred value and gt_depth_value with cv2.text
                    image = cv2.putText(image,f'file: {formatted_file_path}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    image = cv2.putText(image, f'pred: {pred_values_formatted[incremental_file_list]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    image = cv2.putText(image, f'gt: {gt_depth_values[incremental_file_list]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    incremental_file_list += 1
                    image = cv2.imwrite(f'{dataset_folder}/{model_selection}_results/{formatted_file_path}', image)
    last_lr = log_lr_changes(optimizer, last_lr, epoch, epoch_loss)
    

    if epoch % 10 == 0 :
        if show_table:
            # Create a structured table
            epoch_loss_mean = epoch_loss / len(dataset)
            #epoch_loss_mean_formated = "{:.5f}".format(epoch_loss)
            table = [["Epoch", "Epoch Loss", epoch_loss_mean, "", "", "", ""]]
            table.append([epoch, epoch_loss_formatted, "Prediction", "GT", "Error", "D_kpts", "D_mean" ])       
            #print(kpts_list)
            #reshape kpts_list to len(pred_values)
            kpts_list_array = np.array(kpts_list)
            kpts_list_array = kpts_list_array.reshape(len(pred_values),2)
            for i in range(len(pred_values)):
                table.append(["", "", pred_values_formatted[i], gt_depth_values[i],  error_formatted[i], int(kpts_list_array[i][0]),  "{:.3f}".format((kpts_list_array[i][1])/100)])
                
                #table.append(["", "", pred_values_formatted[i], gt_depth_values[i], error_formatted[i], "{:.3f}".format(dic_out['analysis'][i][0]/100),  "{:.3f}".format((dic_out['analysis'][i][1])/100)])
            '''
            print(40 * '-')
            text  = "NEW  EPOCH"
            padding_length = (40 - len(text)) / 2
            padding = '-' * int(padding_length)
            print(padding + text + padding)
            print(40 * '-')
            '''            
            # Print the table
            print(tabulate(table, headers="firstrow", tablefmt="grid"))
        else:
            #print epoch and loss
            #epoch_loss = epoch_loss / len(dataset)
            epoch_loss_formatted = "{:.5f}".format(epoch_loss)
            lr = optimizer.param_groups[0]["lr"]
            lr_formatted = "{:.10f}".format(lr)
            print(f'Epoch: {epoch}, TRAINING loss: {epoch_loss_formatted}, lr: {lr}')
        #l1_loss_from_laplace(pred_depth, gt_depth.to(device),print_loss=True)
        
    
    scheduler.step()#loss)
    #log wandb loss
    wandb.log({"train_loss": epoch_loss}, step=epoch)

def test_loop(dataloader,visual_dataloader):
    if epoch == 1:
        global best_loss
        best_loss = np.inf
        global best_mae
        best_mae = np.inf
    
    pred_values = []
    gt_depth_values = []
    epoch_loss = 0
    kpts_list = []
    incremental_file_list = 0
    gt_depth_values_batch_array = np.array([])
    error_in_prediction_batch_array = np.array([])
    confident_kpts_list = []
    files_list = []


    for i, (file_path, images, depth_values, keypoints, bbox) in enumerate(((dataloader))):
        #print(f'test_i = {i}')
        dic_out = training_prediction(file_path,images, depth_values, keypoints, bbox,status='test')
        kpts_list.append(dic_out['analysis'])
        files_list.append(file_path) #to find hgihest error in prediction

        #duration = time.time()
        samples_data = extract_values(keypoints)
        for i in range(len(samples_data)):
            #print(f'i in sample data = {len(samples_data)}')

            sample = samples_data[i]
            #print()
            #print(f'file: {file_path[i].replace(".txt", "_rgb.png").replace("annotations", "rgb")}')
            #print(np.array(sample['confidence']))
            confident_kpts = (np.array(sample['confidence']) > 0.2).sum().item()
            #print(f'confident about {confident_kpts} keypoints')
            confident_kpts_list.append(confident_kpts)
        #    print()
        #    print(sample)
        #    print()
        #post_duration = time.time() - duration
        #print(post_duration)
        
        #print()
        #print(samples_data)
        #print()
        #print(len(keypoints))
        #print(len(images))
        #for one in range(len(keypoints)):
            #print()
            #print(keypoints[one][2])
            #count_greater_than_0_5 = (keypoints[one][2] > 0.5).sum().item()
            #confident_keypoints
            #print()
            
            #print(images[one].shape)
            #print(keypoints[one][0].shape)
            #confident_keypoints = keypoints[one][0]
        #print(keypoints)
        #print(kpts_list)
        
        gt_depth = depth_values[0]/100
        pred_depth = dic_out['d'].squeeze(1)
        #if 
        #print(f'gt_depth: {gt_depth.flatten()}')
        loss = l1_loss_from_laplace(pred_depth, gt_depth.to(device))
        epoch_loss += loss.item()   
        pred_values.extend(pred_depth.detach().cpu().flatten())
        gt_depth_values.extend(gt_depth.detach().cpu())
        pred_values_array = np.array(pred_values)
        gt_depth_values_array = np.array(gt_depth_values)

        
    # 1) Mean Absolute Error
    mae = mean_absolute_error(gt_depth_values_array, pred_values_array)
    #print(f"Mean Absolute Error: {mae}")

    # 2) Max Absolute Error
    max_error = np.max(np.abs(gt_depth_values_array - pred_values_array))
    #print(f"Max Absolute Error: {max_error}")
    position_error = np.argmax(np.abs(gt_depth_values_array - pred_values_array))
    #print(f"Position of max error: {position_error}")
    
    #print(len(files_list))
    #print(len(files_list))
    
    #images_with_5_max_errors = np.argsort(np.abs(gt_depth_values_array - pred_values_array))[-5:]
    #max_5_postitions = np.argsort(np.abs(gt_depth_values_array - pred_values_array))[-5:]
    

    # 3) Median Absolute Error
    med_error = np.median(np.abs(gt_depth_values_array - pred_values_array))

    wandb.log({"test_loss": epoch_loss, "mae": mae, "med_error": med_error, "max_error": max_error}, step=epoch)

    if mae < best_mae:
        best_loss = epoch_loss
        print(f'Saving model at epoch: {epoch}')
        torch.save(locomodel, f'{test_dataset_folder}/{model_selection}_best_model.pth')
        best_mae = mae
        wandb.log({"best_mae": mae}, step=epoch)
        if epoch > 20: #Just to not lose too much time in the first epochs saving images
            # FInd image with larger error
            image_with_max_error = files_list[np.argmax(np.abs(gt_depth_values_array - pred_values_array))]
            #print(f"Image with max error: {image_with_max_error}")
            # Find 5 images with largest errors
            # Find the absolute differences between ground truth and predicted values
            errors = np.abs(gt_depth_values_array - pred_values_array)

            # Find the indices of the 10 largest errors
            top_10_indices = np.argsort(errors)[-10:]
            #print(top_10_indices)
            # Get the corresponding filenames
            images_with_top_10_errors = [files_list[i] for i in top_10_indices]

            pred_values_formatted = ["{:.2f}".format(value) for value in pred_values_array]
            error = pred_values_array - gt_depth_values_array
            error_formatted = ["{:.2f}".format(value) for value in error]
            epoch_los_formatted = "{:.2f}".format(epoch_loss)
            kpts_list_array = np.array(kpts_list)
            # Flatten the list of lists and extract the first element of each tuple
            auxiliary_data = [x[0].item() for sublist in kpts_list for x in sublist]
            
            # Convert the list to numpy array
            kpts_n_array = np.array(auxiliary_data)

            # Plots area
            error_in_prediction = np.abs(np.array(pred_values_array) - np.array(gt_depth_values_array))
            #print(i,len(gt_depth_values_array),len(error_in_prediction))

            # Create figures for each plot
            fig1, ax1 = plt.subplots()
            ax1.scatter(gt_depth_values_array, error_in_prediction, alpha=0.5)
            ax1.set_title('Error in Prediction vs Ground Truth Depth')
            ax1.set_xlabel('Ground Truth Depth (m)')
            ax1.set_ylabel('Absolute error (m)')
            fig1.savefig(f'{test_dataset_folder}/{model_selection}_abs_error.png', dpi=300)

            fig2, ax2 = plt.subplots()
            ax2.scatter(kpts_n_array, error_in_prediction, alpha=0.5)
            ax2.set_title('Error in Prediction vs Number of Auxiliary Data Points')
            ax2.set_xlabel('Number of Auxiliary Data Points')
            ax2.set_ylabel('Absolute error (m)')
            fig2.savefig(f'{test_dataset_folder}/{model_selection}_kpts.png', dpi=300)

            fig3, ax3 = plt.subplots()
            ax3.scatter(np.array(confident_kpts_list), error_in_prediction, alpha=0.5)
            ax3.set_title('Error in Prediction vs Number of Confident 2D Keypoints')
            ax3.set_xlabel('Number of Confident 2D Keypoints')
            ax3.set_ylabel('Absolute error (m)')
            fig3.savefig(f'{test_dataset_folder}/{model_selection}_AE_with_2D_availability.png', dpi=300)

            fig4, ax4 = plt.subplots()
            # Add plotting code for the fourth plot as needed
            ax4.set_xlabel('Number of Auxiliary Data Points')
            ax4.set_ylabel('Absolute error (m)')
            fig4.savefig(f'{test_dataset_folder}/{model_selection}_AE_with_22D_availability.png', dpi=300)

            # You can close the figures if you're done with them
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
            plt.close(fig4)

            
            #print(np.min(gt_depth_values_array),np.max(gt_depth_values_array))

            #plt.scatter(kpts_n_array, error_in_prediction, alpha=0.5)
            #plt.title('Error in Prediction vs Number of Auxiliary Data Points')
            #plt.xlabel('Number of Auxiliary Data Points')
            #plt.ylabel('Error in Prediction')
            #plt.savefig(f'{test_dataset_folder}/{model_selection}_kpts.png', dpi=300)
            #plt.close()
            # Assuming predicted_depth and groundtruth_depth are numpy arrays

            #print(f"Median Absolute Error: {med_error}")
            
            if save_samples:
                for file_path_file in files_list:
                    formatted_file_path = file_path_file[0].split('/')[-1].replace('.txt', '_rgb.png')
                    image = cv2.imread(f'{test_dataset_folder}/rgb/{formatted_file_path}')
                    #Draw pred value and gt_depth_value with cv2.text
                    cv2.putText(image,f'file: {formatted_file_path}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'pred: {pred_values_formatted[incremental_file_list]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'gt: {gt_depth_values[incremental_file_list]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    incremental_file_list += 1
                    cv2.imwrite(f'{test_dataset_folder}/{model_selection}_results/{formatted_file_path}', image)
                    #print("SAVING IMAGE")

            wandb.log({
                "Error in Prediction vs Ground Truth Depth": wandb.Image(fig1),
                "Error in Prediction vs Number of Auxiliary Data Points": wandb.Image(fig2),
                "Error in Prediction vs Number of Confident 2D Keypoints": wandb.Image(fig3),
                "Fourth Plot (Placeholder)": wandb.Image(fig4),  # Add appropriate title
                "Image example": wandb.Image(f"{test_dataset_folder}/{model_selection}_results/{image_with_max_error[0].split('/')[-1].replace('.txt', '_rgb.png')}",caption='Largest error image'),

            }, step=epoch)
            
            if show_table:
                #epoch_loss_mean = epoch_loss / len(dataset)
                table = [["Epoch", "Epoch Loss", epoch_loss, "", "", "", ""]]
                table.append([epoch, epoch_loss_formatted, "Prediction", "GT", "Error", "D_kpts", "D_mean" ])       

                kpts_list_array = np.array(kpts_list)
                kpts_list_array = kpts_list_array.reshape(len(pred_values),2)
                for i in range(len(pred_values)):
                    table.append(["", "", pred_values_formatted[i], gt_depth_values[i],  error_formatted[i], int(kpts_list_array[i][0]),  "{:.3f}".format((kpts_list_array[i][1])/100)])

                # Print the table
                print(tabulate(table, headers="firstrow", tablefmt="grid"))
            else:
                #print epoch and loss
                #epoch_loss = epoch_loss / len(dataset)
                epoch_loss_formatted = "{:.5f}".format(epoch_loss)
                print(f'Epoch: {epoch}, TESTING loss: {epoch_loss_formatted}')
            
            if epoch == epochs:
                print("Images with top 10 errors:", images_with_top_10_errors)

def visual_loop(dataloader):
    print('Visualizing results...')
    start_time = time.time()
    # Camera parameters for RealSense L515
    focal_length_mm = 3.5  # Focal length in millimeters
    sensor_width_mm = 4.2  # Sensor width in millimeters
    sensor_height_mm = 4.2  # Sensor height in millimeters

    # Camera image size
    image_width = 640
    image_height = 360

    # Calculate the focal length in pixels
    focal_length_pixels = (focal_length_mm / sensor_width_mm) * image_width

    # Camera intrinsic matrix
    fx = fy = focal_length_pixels
    cx = image_width / 2
    cy = image_height / 2

    # Calculate horizontal and vertical FOV in radians
    fov_x_rad = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
    fov_y_rad = 2 * np.arctan(sensor_height_mm / (2 * focal_length_mm))

    # Calculate the maximum x and y at the maximum depth for the FOV lines
    max_depth = 10
    max_x = max_depth * np.tan(fov_x_rad / 2)
    max_y = max_depth * np.tan(fov_y_rad / 2)

    # Create figure and axes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # 
    black_image_array = np.zeros((int(image_height*1.5), int(image_width*1.5), 3), dtype=np.uint8)
    #img = imread(f'test_images/test_1.png')

    axs[1].imshow(black_image_array)
    axs[1].axis('off')  # Hide axes

    # Plot settings for the first subplot
    axs[0].set_title("Top-Down View Map")
    axs[0].set_xlabel("X (left-right distance from camera, meters)")
    axs[0].set_ylabel("Y (distance from camera, meters)")
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(0, 10)

    # Create a Rectangle patch (field of view)
    #print(max_depth, max_x, max_y)
    fov = patches.Polygon(((0, 0), (-max_x, max_depth), (max_x, max_depth)), closed=True, color='blue', alpha=0.2)
    axs[0].add_patch(fov)

    # Object's initial pixel position and depth value
    x_pixel = 320
    y_pixel = 240
    depth = 2.0

    # Initialize scatter
    scat = axs[0].scatter([], [])
    #plt.tight_layout()
    #plt.show()
    for i, (file_path, images, depth_values, keypoints, bbox) in enumerate(((dataloader))):
        dic_out = training_prediction(file_path,images, depth_values, keypoints, bbox, status='visual')
        
        pred_depth = dic_out['d'].squeeze(1).detach().cpu()
        
        #pred_value = pred_depth.cpu().detach().numpy()
        #print(file_path)

        file = file_path[0].split('/')[-1].replace('.txt', '')
        numpy_image = [tensor.numpy() for tensor in images[0]]
        numpy_list = [tensor.numpy() for tensor in bbox[0]]
        numpy_image = cv2.cvtColor(np.array(numpy_image).transpose(1,2,0), cv2.COLOR_RGB2BGR)

        # Convert the Python list of NumPy arrays into a NumPy array
        bounding_box = np.array(numpy_list)

        # Calculate the midpoint along each axis
        midpoint_x = (bounding_box[0, 0] + bounding_box[2, 0]) / 2
        midpoint_y = (bounding_box[1, 0] + bounding_box[3, 0]) / 2

        # Create a new 2x1 NumPy array representing the midpoint
        
        midpoint = np.array([[midpoint_x], [midpoint_y]])
        #Visualize image with bounding box and midpoint
        
        gt_depth = depth_values[0]/100    
        
        cv2.rectangle(numpy_image, (int(bounding_box[0, 0]), int(bounding_box[1, 0])), (int(bounding_box[2, 0]), int(bounding_box[3, 0])), (0, 255, 0), 2)
        cv2.circle(numpy_image, (int(midpoint[0, 0]), int(midpoint[1, 0])), 10, (0, 0, 255), -1)
        

        x_pixel = midpoint_x  # Moving the object from left to right

        # Convert pixel coordinates to camera coordinates
        Xc = (x_pixel - cx) * pred_depth / fx
        Xc2 = (x_pixel - cx) * gt_depth / fx

        #Add the predicted depth value and the groundtruth value to the scatter plot
        # Add the predicted depth value and the groundtruth value to the scatter plot
        scat_pred = axs[0].scatter(Xc, pred_depth, color='blue', label='Predicted depth',s= 500)
        
        scat_gt = axs[0].scatter(Xc2, gt_depth, color='red', label='Groundtruth depth',s= 400)


        # Load image
        #img = imread(f'{test_dataset_folder}/rgb/{file}_rgb.png')
        #print(images.squeeze(0).shape)
        #convert tensor to numpy array
        #img = images.squeeze(0).permute(1,2,0).numpy()
        #numpy_image = numpy_image.astype('uint8')
        #numpy_image = cv2.resize(numpy_image, (int(image_width*0.5), int(image_height*0.5)))
        axs[1].imshow(cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB))
        axs[1].axis('off')  # Hide axes
        #plt.imsave(img, name)
        plt.savefig(f'{test_dataset_folder}/{model_selection}_visual/{file}_v.png', dpi=300)

        #fig = plt.gcf()  # Get the current figure
        #fig.canvas.draw()  # Draw the figure on the canvas
        #width, height = fig.canvas.get_width_height()
        #img = Image.frombuffer('RGB', (width, height), fig.canvas.tostring_rgb())
        #img.save(f'{test_dataset_folder}/{model_selection}_visual/{file}_v.png')
        #plt.savefig(f'{test_dataset_folder}/{model_selection}_visual/{file}_v.png', dpi=300)
        # Remove the scatter plot points
        scat_pred.remove()
        scat_gt.remove()
    print('Visualizing results done...')


print("Started logging in wandb")
wandb_config = OmegaConf.to_container(opt, resolve=True, throw_on_missing=True)
#wandb.init(project=opt.wandb.project_name,entity=opt.wandb.entity,
#    name='{}_{}_{}_{}_{}'.format(opt.data.dataset,opt.network.task,opt.network.weight,opt.network.archit,opt.network.grad_method),
#    config = wandb_config)
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="socialy-aware-AMRs",
    name=f"{model_selection}",
    # track hyperparameters and run metadata
    config= wandb_config,
)

wandb.log({"Model architecture": wandb.Image(f'model_{model_selection}.gv.png'), "Auxmodel architecture": wandb.Image(f'auxmodel_{model_selection}.gv.png')})



print('Starting training')

for epoch in range(1,epochs+1):
    print(f"Epoch {epoch}/{epochs}")
    train_loop(dataloader,last_lr)
    test_loop(test_dataloader,visual_dataloader)
    
# Visualizing images, but need to fix this
#visual_loop(visual_dataloader)


# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
    
    

#Info table of training:
print(40 * '-')
print("Training Info")
print(40 * '-')
print(f"Model: {model_selection}")
print(f"Dataset: {dataset_folder}")
print(f"Dataset length: {len(dataset)}")
print(f"Batch_size: {batch_size}")
print(f"Optimizer: {optimizer}")
print(f"Scheduler: {scheduler}")
print(f"Learning rate list changes: {lr_list}")