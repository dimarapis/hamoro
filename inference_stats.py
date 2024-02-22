import torch
import numpy
from ultralytics import YOLO
from humaNN_model import Loco
import numpy as np
from tqdm import tqdm
#pose_model = YOLO('yolov8l-pose.pt')  # load a pretrained model (recommended for training)
#seg_model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
#locomodel = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net='two_stage', device=torch.device("cuda"), n_dropout = 0, p_dropout=0, num_stage=3)

def timing_step(model_type):
    
    device = torch.device("cuda")
    dummy_input_rgb = torch.randn(1, 3,640,480, dtype=torch.float).to(device)
    #dummy_input_depth = torch.randn(1, 1,640,480, dtype=torch.float).to(device)
    dummy_inpuy_kpts = torch.randn(1, 4, 17, dtype=torch.float).to(device)
    dummy_input_bbox = torch.rand(1, 4, dtype=torch.float).to(device)
    K_original = [[599.978, 0.0, 318.604], [0.0, 600.5, 247.77], [0.0, 0.0, 1.0]]

    if model_type == 'pose':
        model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)
        model.to(device)
        inputs = dummy_input_rgb
        model_parameters = sum(p.numel() for p in model.model.parameters())
        print(f'Number of parameters: {model_parameters/1000000}')
        #remove last layer
        print(model.model)
        model_parameters = sum(p.numel() for p in model.model.parameters())
        
        #check paramers of specific layer
        #layer = model.model[-1]
        
        print(f'Number of parameters: {model_parameters/1000000}')
    elif model_type == 'seg':
        model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
        model.to(device)
        inputs = dummy_input_rgb
        model_parameters = sum(p.numel() for p in model.model.parameters())
        
        print(f'Number of parameters: {model_parameters/1000000}')

    elif model_type == 'kpts':

        model = Loco(model = 'monoloco-190719-0923.pkl', mode='mono', net='two_stage', device=torch.device("cuda"), n_dropout = 0, p_dropout=0, num_stage=3)

        model_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        aux_model_parameters = sum(p.numel() for p in model.auxmodel.parameters() if p.requires_grad)
        print(f'Number of parameters: {(model_parameters + aux_model_parameters)/1000000}')

    elif model_type == 'posesolo':
        from ultralytics.nn.modules import Pose
        model = Pose(nc=1,ch=(128, 256, 512))
        model.to(device)
        model_parameters = sum(p.numel() for p in model.parameters())
        
        print(f'Number of parameters: {model_parameters/1000000}')

    elif model_type == 'segsolo':
        from ultralytics.nn.modules import Segment
        model = Segment(nc=1,ch=(128, 256, 512))
        model.to(device)
        model_parameters = sum(p.numel() for p in model.parameters())
        
        print(f'Number of parameters: {model_parameters/1000000}')
    

    # INIT LOGGERS
    
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        if model_type == 'kpts':
            _ = model.forward(dummy_inpuy_kpts, K_original,dummy_input_bbox,'test')
        else:
            _ = model(inputs)
            
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            if model_type == 'kpts':
                _ = model.forward(dummy_inpuy_kpts, K_original,dummy_input_bbox,'test')
            else:
                _ = model(inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn, std_syn)
    


timing_step('seg')