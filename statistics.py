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
import matplotlib.ticker as tck

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
    


model_1 = 'monoloco'
model_2 = 'two_stage'
test_dataset_folder = 'data/dimidata_test_filtered_fixed/'
train_dataset_folder = 'data/dimidata_train_filtered'
num_stage = 3



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

def training_prediction(model,filepath, image, depth, keypoints, bbox,status):
    tensor_data = [torch.stack(item) for item in keypoints]
    tensor_data = torch.stack(tensor_data)
    tensor_data = tensor_data.permute(2,1,0)
    #print(tensor_data.shape)
    #print(tensor_data)
    dic_out = model.forward(tensor_data, K_original,bbox,status)
    #print(dic_out)
    dic_out['analysis'] = analysis(tensor_data)

    return dic_out


#locomodel = Loco2(mode='mono', net='monoloco', device=device, n_dropout = 0, p_dropout=0)
#locomodel = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net='monoloco', device=device, n_dropout = 0, p_dropout=0)

#model_1 = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net=model_1, device=device, n_dropout = 0, p_dropout=0, num_stage=num_stage)
#model_2 = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net=model_2, device=device, n_dropout = 0, p_dropout=0, num_stage=num_stage)


#model_selection = 'monoloco_trained'
#locomodel = torch.load(f'{test_dataset_folder}/{model_selection}_best_model.pth')
#locomodel.load(f'{test_dataset_folder}/{model_selection}_best_model.pth')

test_dataset = DepthDataset(test_dataset_folder)
K_original = [[599.978, 0.0, 318.604], [0.0, 600.5, 247.77], [0.0, 0.0, 1.0]]


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def statistics_loop(dataloader):
    
    for model in ['monoloco','two_stage']:
        if model == 'monoloco':
            model_name = 'Monoloco'
        else:
            model_name = 'Our model'
        #model_1 = torch.load(f'best_models/{model_1}_best_model.pth')
        torch_model = torch.load(f'best_models/{model}_best_model.pth')
        pred_values,pred_values_0m,pred_values_3m = [],[],[]
        gt_depth_values,gt_depth_values_0m,gt_depth_values_3m = [],[],[]
        kpts_list = []
        confident_kpts_list = []
        files_list = []
        for i, (file_path, images, depth_values, keypoints, bbox) in enumerate(((dataloader))):
            #print(f'test_i = {i}')
            dic_out = training_prediction(torch_model,file_path,images, depth_values, keypoints, bbox,status='test')
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
            
            if gt_depth <= 3:
                pred_values_0m.extend(pred_depth.detach().cpu().flatten())
                gt_depth_values_0m.extend(gt_depth.detach().cpu())
            #if 
            
            else:
                pred_values_3m.extend(pred_depth.detach().cpu().flatten())
                gt_depth_values_3m.extend(gt_depth.detach().cpu())
            #print(f'gt_depth: {gt_depth.flatten()}')
            pred_values.extend(pred_depth.detach().cpu().flatten())
            gt_depth_values.extend(gt_depth.detach().cpu())
            
        
        
        pred_values_array = np.array(pred_values)
        gt_depth_values_array = np.array(gt_depth_values)
        pred_values_0m = np.array(pred_values_0m)
        gt_depth_values_0m = np.array(gt_depth_values_0m)
        pred_depth_3m = np.array(pred_values_3m)
        gt_depth_values_3m = np.array(gt_depth_values_3m)   
            
        # 1) Mean Absolute Error
        mae = mean_absolute_error(gt_depth_values_array, pred_values_array)
        #print(f"Mean Absolute Error: {mae}")

        # 2) Max Absolute Error
        max_error = np.max(np.abs(gt_depth_values_array - pred_values_array))
        #print(f"Max Absolute Error: {max_error}")
        
        mae_0m = mean_absolute_error(gt_depth_values_0m, pred_values_0m)
        max_error_0m = np.max(np.abs(gt_depth_values_0m - pred_values_0m))
        mae_3m = mean_absolute_error(gt_depth_values_3m, pred_values_3m)
        max_error_3m = np.max(np.abs(gt_depth_values_3m - pred_values_3m))
        print(f'mae: {mae}, max_error: {max_error}, len(pred_values): {len(pred_values)}')
        print(f'MAE 0m: {mae_0m}, MAE 3m: {mae_3m}',f'Max error 0m: {max_error_0m}, Max error 3m: {max_error_3m},len(pred_values_0m): {len(pred_values_0m)}, len(pred_values_3m): {len(pred_values_3m)}')
        #len(pred_values_3m)
        #len(pred_values)
        
        #print(f"Max Absolute Error: {max_error}")
        position_error = np.argmax(np.abs(gt_depth_values_array - pred_values_array))
        #print(f"Position of max error: {position_error}")
        
        #print(len(files_list))
        #print(len(files_list))
        
        #images_with_5_max_errors = np.argsort(np.abs(gt_depth_values_array - pred_values_array))[-5:]
        #max_5_postitions = np.argsort(np.abs(gt_depth_values_array - pred_values_array))[-5:]
        

        # 3) Median Absolute Error
        med_error = np.median(np.abs(gt_depth_values_array - pred_values_array))


    
            
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
        ax1.set_title(f'{model_name} prediction error with distance')
        ax1.set_xlim(0, 5)
        ax1.set_ylim(-0.1, 2.5)
        ax1.set_xlabel('Ground Truth Depth (m)')
        ax1.set_ylabel('Absolute error (m)')
        fig1.savefig(f'plots/statistics/{model}_abs_error.png', dpi=300)

        fig2, ax2 = plt.subplots()
        ax2.scatter(kpts_n_array, error_in_prediction, alpha=0.5)
        ax2.set_title(f'{model_name} prediction error with auxiliary data points')
        ax2.set_xlim(-0.5, 17.5)
        ax2.xaxis.set_major_locator(tck.MultipleLocator())
        ax2.set_ylim(-0.1, 2.5)
        ax2.set_xlabel('Number of Auxiliary Data Points')
        ax2.set_ylabel('Absolute error (m)')
        fig2.savefig(f'plots/statistics/{model}_kpts.png', dpi=300)

        fig3, ax3 = plt.subplots()
        ax3.scatter(np.array(confident_kpts_list), error_in_prediction, alpha=0.5)
        ax3.set_title(f'{model_name} prediction error with 2D Keypoints')
        ax3.set_xlim(-0.5, 17.5)
        ax3.xaxis.set_major_locator(tck.MultipleLocator())
        ax3.set_ylim(-0.1, 2.5)
        ax3.set_xlabel('Number of 2D Keypoints')
        ax3.set_ylabel('Absolute error (m)')
        fig3.savefig(f'plots/statistics/{model}_AE_with_2D_availability.png', dpi=300)

        fig4, ax4 = plt.subplots()
        # Add plotting code for the fourth plot as needed
        ax4.scatter(np.array(gt_depth_values_array), pred_values_array, alpha=0.5)
        #find line of best fit
        line_model = np.polyfit(gt_depth_values_array, pred_values_array, 1)
        
        #ADD LINE 
        predict = np.poly1d(line_model)
        from sklearn.metrics import r2_score
        r_squared = r2_score(pred_values_array, predict(gt_depth_values_array)).round(3)
        print(r_squared)
        print(line_model)
        # Write r_quared on plit
        ax4.text(0.65, 0.1, f'y = {line_model[0].round(3)}x + {line_model[1].round(3)}', horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes, color='black')
        ax4.text(0.65, 0.05, f'R^2 = {r_squared}', horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes, color='black')
        #add line model to plot
        #ax4.text(0.85, 0.05, f' = {len(gt_depth_values_array)}', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, color='black'
        #add line of best fit to plot
        ax4.plot(gt_depth_values_array, line_model[0]*gt_depth_values_array+line_model[1], color='black')
        ax4.set_xlim(0, 5)
        ax4.set_ylim(-0, 5)

        ax4.set_title(f'{model_name}')
        ax4.set_xlabel('Ground truth distance (m)')
        ax4.set_ylabel('Predicted distance (m)')
        fig4.savefig(f'plots/statistics/{model}_pred_gt.png', dpi=300)

        # You can close the figures if you're done with them
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        
        mu = np.mean(error_in_prediction)
        sigma = np.std(error_in_prediction)
        
        print(f'{model}: Mean Absolute Error: {mu}, Standard Deviation: {sigma}')
        


    
statistics_loop(test_dataloader)