import torch
import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex

from dataloader_eh import FloorDataset
import numpy as np
import time
import cv2

def compute_recall(predicted_batch, ground_truth_batch, num_classes):
    recalls = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        true_positive = np.sum(np.logical_and(predicted_batch == class_idx, ground_truth_batch == class_idx))
        false_negative = np.sum(np.logical_and(predicted_batch != class_idx, ground_truth_batch == class_idx))
        
        if true_positive + false_negative > 0:
            recalls[class_idx] = true_positive / (true_positive + false_negative)
        else:
            recalls[class_idx] = 0.0
            
    return recalls


def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def binary_recall(true_labels, predicted_labels):
    true_positive = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
    false_negative = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
    
    recall = calculate_recall(true_positive, false_negative)
    return recall



def data_loader(dataset, batch_size=0, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)    

def generate_tensor(batch_size):
    tensor_shape = (batch_size, 256, 256)
    tensor = torch.zeros(tensor_shape,requires_grad=True)
    #tensor[random_checker(tensor_shape).gt(0.5)] = 1
    return tensor

def get_bboxes_etc(result):
        height,width = 256,256

        segmentation_contours_idx = []
        if result.masks is not None:
            
            for seg in result.masks.xy:
                # contours
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)

            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            # Get class ids
            class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
            # Get scores
            scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
            
            masks = result.masks
            
            return bboxes, class_ids, segmentation_contours_idx, scores, masks
        else:
            return None, None, None, None

def binary_cross_entropy_loss(outputs, targets):
    # Ensure the inputs are of type float and targets are of type long
    outputs = outputs.float()
    targets = targets.float()

    # Calculate binary cross-entropy loss
    bce_loss = nn.BCELoss()
    loss = bce_loss(outputs, targets)
    return loss

def plot_samples_color(val_dataloader, device, model, metric):
    recalls = []
    for i, (inputs, gt) in enumerate(tqdm(val_dataloader)):
        #for inputs, gt in val_dataloader:
        #for inputs, gt in val_dataloader:
        inputs, gt = inputs.to(device), gt.to(device)
        with torch.no_grad():
            # Forward pass
            _, outputs = model(inputs.to(device))
        #print(outputs.shape)
        #print(gt.shape)
        normalized_gt = gt.type(torch.cuda.FloatTensor)
        

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # Reshape mean and standard deviation to (1, c, 1, 1)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)


        unnormalized_images = inputs * std + mean

        images = [transforms.ToPILImage()(img) for img in unnormalized_images]
        normalized_gt = normalized_gt.int() * 255
        normalized_gt = [transforms.ToPILImage()(mask.squeeze()) for mask in normalized_gt]
        outputs = torch.round(outputs)
        outputs = outputs.int() * 255
        
        outputs = [transforms.ToPILImage()(pred.squeeze()) for pred in outputs]
        
        plt.close()
        #print(f'lol{i}')
        # Create a 2x4 plot
        for x in range(0, 4):
            #print(outputs[0])
            #print(normalized_gt[0].shape)
            #print(images[0].shape)
            #print(np.max(outputs[0]))

            
            # Convert the arrays to the expected data type (uint8)
            predicted_image = np.array(outputs[x], dtype=np.uint8)
            ground_truth_image = np.array(normalized_gt[x], dtype=np.uint8)
            image = np.array(images[x], dtype=np.uint8)
            
            predicted_image = np.stack((predicted_image,) * 3, axis=-1)
            ground_truth_image = np.stack((ground_truth_image,) * 3, axis=-1)

            
            #print(predicted_image.shape)
            #print(ground_truth_image.shape)
            
            # Create grayscale versions of the images for comparison
            predicted_grayscale = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY)
            ground_truth_grayscale = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

            #print(predicted_image.shape)
            #print(ground_truth_image.shape)
            
            
            # Convert grayscale images to RGB format
            #predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_GRAY2RGB)
            #ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_GRAY2RGB)

            
            visualization_image = image.copy()#np.zeros_like(predicted_image, dtype=np.uint8)

            # Create masks for different conditions
            both_agree_mask = np.logical_and(predicted_grayscale == 255, ground_truth_grayscale == 255)
            predicted_mask = np.logical_and(predicted_grayscale == 255, ground_truth_grayscale == 0)
            ground_truth_mask = np.logical_and(predicted_grayscale == 0, ground_truth_grayscale == 255)

            
            both_agree_obst = np.logical_and(predicted_grayscale == 0, ground_truth_grayscale == 0)
            predicted_obst_mask = np.logical_and(predicted_grayscale == 255, ground_truth_grayscale == 0)
            ground_truth_obst_mask = np.logical_and(predicted_grayscale == 0, ground_truth_grayscale == 255)

            
            #count amount of pixels
            num_both_agree_obst = np.count_nonzero(both_agree_obst)
            num_predicted_obst = np.count_nonzero(predicted_obst_mask)
            numb_ground_truth_obst = np.count_nonzero(ground_truth_obst_mask)
            #print(f'both_agree: {num_both_agree_obst}')
            #print(f'pred obst: {num_predicted_obst}')
            #print(f'numb_ground_truth_obst: {numb_ground_truth_obst}')
            #print(numb_ground_truth_obst)
            #print(np.sum(num_both_agree_obst, num_predicted_obst, numb_ground_truth_obst))            
            obstacleRecall = num_both_agree_obst / (num_both_agree_obst + numb_ground_truth_obst)
            #print(f'obstacleRecall: {obstacleRecall}')
            # Set pixel values based on masks

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            visualization_image = image.copy()#np.zeros_like(predicted_image, dtype=np.uint8)
            visualization_image[both_agree_mask] = [211, 211, 211]   # Gray
            visualization_image[predicted_mask] = [55, 61, 163]#[181, 54, 34]   # Red
            visualization_image[ground_truth_mask] = [55, 115, 55] # Blue
            
            
            #visualization_image = cv2.cvtColor(visualization_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'plots/floor_visual/{i}_{x}.png', visualization_image)
            cv2.imwrite(f'plots/floor_visual/{i}_{x}_rgb.png', image)
            cv2.imwrite(f'plots/floor_visual/{i}_{x}_pred.png', predicted_grayscale)
            
            recalls.append(obstacleRecall)
    recalls = np.array(recalls)
    #print(recalls)
    recals = np.mean(recalls).round(4)
    #print(f'ObstacleRec: {recals}')
def plot_samples_trio(images,masks,preds,batch_size):
    # Convert PyTorch tensors back to PIL Images for plotting
    
    images = [transforms.ToPILImage()(img) for img in images]
    masks = masks.int() * 255
    masks = [transforms.ToPILImage()(mask.squeeze()) for mask in masks]
    preds = torch.round(preds)
    preds = preds.int() * 255
    preds = [transforms.ToPILImage()(pred.squeeze()) for pred in preds]
    
    plt.close()
    # Create a 2x4 plot
    fig, axes = plt.subplots(3, int(batch_size), figsize=(18, 6))

    # Plot the images and corresponding ground truth images
    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        if  i < 8:
            axes[0, i].imshow(image)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Image {i+1}")

            axes[1, i].imshow(mask, cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Ground Truth {i+1}")
            
            axes[2, i].imshow(pred, cmap="gray")
            axes[2, i].axis("off")
            axes[2, i].set_title(f"Prediction {i+1}")
            
        else:
            i = i-8
            axes[2, i].imshow(image)
            axes[2, i].axis("off")
            axes[2, i].set_title(f"Image {i+1}")

            axes[3, i].imshow(mask, cmap="gray")
            axes[3, i].axis("off")
            axes[3, i].set_title(f"Ground Truth {i+1}")
            
    plt.tight_layout()
    plt.savefig('floor_example.png')
    #plt.show()

def validation_step(val_dataloader, device, model, metric):
    #inputs, gt = next(iter(val_dataloader))
    # Create empty array
    mIoU = []
    for inputs, gt in tqdm(val_dataloader):
        #for inputs, gt in val_dataloader:
        #for inputs, gt in val_dataloader:
        inputs, gt = inputs.to(device), gt.to(device)
        with torch.no_grad():
            # Forward pass
            _, outputs = model(inputs.to(device))
        #print(outputs.shape)
        #print(gt.shape)
        normalized_gt = gt.type(torch.cuda.FloatTensor)
        

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # Reshape mean and standard deviation to (1, c, 1, 1)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)


        outputs_0_1 = torch.round(outputs)
        #print(outputs_0_1)
        #print(normalized_gt)
        IoU = metric(outputs_0_1, normalized_gt)
        #Coimpute recall for each class
        #recalls = compute_recall(outputs, normalized_gt)
        mIoU.append(IoU.item())    
        
        #r#ecall = binary_recall(outputs_0_1, normalized_gt)

        #print(recall)
    #print(mIoU)
    mIoU = np.asarray(mIoU)
    #print(mIoU)
    #print(msIoU)
    mIoU = np.mean(mIoU).round(4)
    print(f'mIoU: {np.mean(mIoU).round(4)}')
    #print("Recall:", recall)

    #meanIoU = torch.mean(torch.tensor(mIoU))
    #print(f' mIoU: {meanIoU.item()}')
    print(20*'=')
    # Unnormalize the tensor
    unnormalized_images = inputs * std + mean
    return mIoU
    #plot_samples_trio(unnormalized_images, normalized_gt, outputs, batch_size=4)

def timing_step(val_dataloader, device, model, metric):
    inputs, gt = next(iter(val_dataloader))
    # Create empty array
    #mIoU = []

    device = torch.device("cuda")
    model.to(device)
    #dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _,_outputs = model(inputs.to(device))
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(inputs.to(device))
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    #print(mean_syn, std_syn)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    metric = BinaryJaccardIndex().to(device)
    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor()
    ])

    image_dir = "data/floor/all_mixed/rgb"
    mask_dir = "data/floor/all_mixed/mask"
    train_dataset = FloorDataset(image_dir, mask_dir,transform=data_transforms)
    val_dataset = FloorDataset(image_dir.replace('all_mixed','24k'), mask_dir.replace('all_mixed','24k'),transform=data_transforms)

    # Create the DataLoader

    train_dataloader = data_loader(dataset=train_dataset, batch_size=batch_size, shuffle = False, num_workers = 0)
    val_dataloader = data_loader(dataset=val_dataset, batch_size=4, shuffle = False, num_workers = 0)

    #print(f"Length of dataset: {len(train_dataset)}")

    #device = "cpu"
    from multi_social_aware_perception import RebuiltYOLO, RebuiltYOLOLarge

    from ultralytics import YOLO
    
    #seg_model = YOLO('yolov8l-seg.pt')  # load an official model

    #print(seg_model.model)
    model = RebuiltYOLO().to(device)
    #torch_model = torch.load('yolov8s-seg.pt'),
    saved_state_dict = torch.load('yolov8s-seg.pt')['model'].state_dict()
    
    updated_state_dict = {}
    for layer_name, weights in saved_state_dict.items():
        updated_layer_name = '.'.join(layer_name.split('.')[1:])
        if updated_layer_name in ['22.cv3.0.2.weight', '22.cv3.0.2.bias', '22.cv3.1.2.weight', '22.cv3.1.2.bias', '22.cv3.2.2.weight', '22.cv3.2.2.bias']:
            continue
        else:
            updated_state_dict[updated_layer_name] = weights

    model.load_state_dict(updated_state_dict, strict=False)


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100
    best_mIoU = 0
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, gt in tqdm(train_dataloader):
            inputs, gt = inputs.to(device), gt.to(device)
            optimizer.zero_grad()

            # Forward pass
            _, outputs = model(inputs.to(device))

            normalized_gt = gt.type(torch.cuda.FloatTensor)
            loss = criterion(outputs, normalized_gt)

            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        scheduler.step()

        # Calculate average loss and metrics
        epoch_loss = running_loss / len(train_dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.4f}")

        mIoU = validation_step(val_dataloader, device, model, metric)
        print(f'mIoU: {mIoU}')
        if mIoU > best_mIoU:
            
        #timing_step(val_dataloader, device, model, metric)
        
            plot_samples_color(val_dataloader, device, model, metric)


            torch.save(model.state_dict(), 'best_floor_model.pt')
'''
def infer_folder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.PILToTensor()
        ])
    transform_RGB =   transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                                   transforms.Resize((256,256)),
                                                   transforms.ToTensor()])
    
    model = SegmentationModel().to(device)#.segmentation_model()
    model.load_state_dict(torch.load('best_deeplab_model.pt'))    
    
    image_list = os.listdir('warehouse_nn')
    for image_path in image_list:
        image_pil = Image.open(f'warehouse_nn/{image_path}').convert("RGB")
        image_shape = image_pil.size
        image = transform_RGB(image_pil).float()
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image = image.unsqueeze(0)
        outputs = model(image.to(device))
        
        preds = torch.round(outputs)
        #preds = preds.int() * 255
        pred = transforms.ToPILImage()(preds.squeeze())
        pred = pred.resize(image_shape)
        #pred = preds[0].squeeze(0).detach().cpu().numpy()
        #im = Image.fromarray(pred)
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Display the grayscale image in the first subplot
        ax1.imshow(pred, cmap='gray')
        ax1.set_title('Grayscale Image')

        # Display the RGB image in the second subplot
        ax2.imshow(image_pil)
        ax2.set_title('RGB Image')

        # Save the plot as a file
        plt.savefig('warehouse_nn_save/'+image_path.split('.')[0]+'_pred.png')
        #pred.save('warehouse_nn/'+image_path.split('.')[0]+'_pred.png')
#infer_folder()
'''
train()