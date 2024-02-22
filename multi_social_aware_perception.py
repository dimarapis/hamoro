import torch.nn as nn
import torch
from ultralytics.nn.modules import Pose, Segment
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import logging
from PIL import Image, ImageDraw

from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
import numpy as np
import matplotlib.patches as patches
import pycocotools.coco 
import pycocotools.cocoeval
import os
import json
from tqdm import tqdm
from ultralytics.yolo.engine.results import Results
from ultralytics import YOLO
import torch.nn.functional as F

def yolo_seg_example():
    pass
name_dict = {
0:'person',
1:'bicycle',
2:'car',
3:'motorcycle',
4:'airplane',
5:'bus',
6:'train',
7:'truck',
8: 'boat',
9: 'traffic light',
10: 'fire hydrant',
11: 'stop sign',
12:'parking meter',
13:'bench',
14:'bird',
15:'cat',
16:'dog',
17:'horse',
18:'sheep',
19:'cow',
20:'elephant',
21:'bear',
22:'zebra',
23:'giraffe',
24:'backpack',
25:'umbrella',
26:'handbag',
27:'tie',
28:'suitcase',
29:'frisbee',
30:'skis',
31:'snowboard',
32:'sports ball',
33:'kite',
34:'baseball bat',
35:'baseball glove',
36:'skateboard',
37:'surfboard',
38:'tennis racket',
39:'bottle',
40:'wine glass',
41:'cup',
42:'fork',
43:'knife',
44:'spoon',
45:'bowl',
46:'banana',
47:'apple',
48:'sandwich',
49:'orange',
50:'broccoli',
51:'carrot',
52:'hot dog',
53:'pizza',
54:'donut',
55:'cake',
56:'chair',
57:'couch',
58:'potted plant',
59:'bed',
60:'dining table',
61:'toilet',
62:'tv',
63:'laptop',
64:'mouse',
65:'remote',
66:'keyboard',
67:'cell phone',
68:'microwave',
69:'oven',
70:'toaster',
71:'sink',
72:'refrigerator',
73:'book',
74:'clock',
75:'vase',
76:'scissors',
77:'teddy bear',
78:'hair drier',
79:'toothbrush'
}

def seg_postprocess(preds, img, orig_imgs, path):
    
    name_dict = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8: 'boat',9: 'traffic light',10: 'fire hydrant',
11: 'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',21:'bear',22:'zebra',23:'giraffe',
24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',
35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',
46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'couch',
58:'potted plant',59:'bed',60:'dining table',61:'toilet',62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',
69:'oven',70:'toaster',71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush'}

    """Apply Non-maximum suppression to prediction outputs."""
    p = ops.non_max_suppression(preds[0],
                                0.25,
                                0.7,
                                agnostic=False,
                                max_det=300,
                                nc=80,
                                classes=None)
    
    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        #img = torch.zeros(1,3,720,1280)
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        orig_img = np.array(orig_img)
        img_path = path[i] if isinstance(path, list) else path
        masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
        if not isinstance(orig_imgs, torch.Tensor):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names=name_dict, boxes=pred[:, :6], masks=masks))
    return results
       
def yolo_pose_example():
    # Load image
    img_path_1 = 'bus.jpg'#/test_24a_2.png'
    img_path_2 = 'test/test_24a_2.png'
    image_1 = Image.open(img_path_1).convert('RGB')#.resize(IMAGE_SHAPE)
    image_2 = Image.open(img_path_2).convert('RGB')#.resize(IMAGE_SHAPE)
    processed_image_1 = transform(image_1).to('cuda:0').half()
    processed_image_2 = transform(image_2).to('cuda:0').half()
    double_tensor = torch.stack([processed_image_1, processed_image_2], dim=0)

    with torch.no_grad():
        # Forward pass and postprocess predictions
        output = multimodel(double_tensor)#.unsqueeze(0))
        final_output = postprocess(output[0])
        
    json_dict = {'annotations': []}
    json_dict = output_to_json_2(json_dict['annotations'],final_output)
    #print(json_dict)
    visualize(final_output, image_1.resize((640,640)), image_2.resize((640,640)))#.transpose()

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
def postprocess(preds):
    """Postprocesses predictions and returns a list of Results objects."""
    preds = ops.non_max_suppression(preds,
                                    conf_thres=0.25,
                                    iou_thres=0.7,
                                    agnostic=False,
                                    max_det=300,
                                    classes=None,
                                    nc=1)
    return preds

def visualize_cv2():
    cv2_image = cv2.imread('test/test_24a_2.png')
    cv2_image = cv2.resize(cv2_image, (640,640))
    cv2.rectangle(cv2_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # draw a green rectangle with thickness of 2
    cv2.imshow('image', cv2_image)
    cv2.waitKey(0)

def output_to_json_2(json_dict, final_output):
    for i, batch_item in enumerate(final_output):
        #print(len(batch_item))
        for pred in batch_item:
        #print(pred)
            json_pred = output_to_json(pred,i)
            json_dict.append(json_pred)
        
        #print(json_dict)
        #json_dict = {'annotations': json_pred}
    return json_dict

def visualize(final_output, image_1, image_2):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    #print(image_1)
    ax[0].imshow(image_1.copy())
    ax[0].set_title('Image_1')
    ax[1].imshow(image_2.copy())
    ax[1].set_title('Image_2')
    
    #Keypoints format
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
        [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
    ]  
    keypoint_color = (255, 0, 0)  # red
    skeleton_color = (0, 255, 0)  # green
    
    for i, batch_item in enumerate(final_output):
        json_dict = {'annotations':[]}
        #print(len(batch_item))
        for pred in batch_item:
        #print(pred)
            json_pred = output_to_json(pred,i)
            json_dict['annotations'].append(json_pred)
            bbox = json_pred['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Create a rectangle patch with the bounding box coordinates
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

            # Add the rectangle patch to the plot   
            ax[i].add_patch(rect)


            keypoints = json_pred['keypoints']


            keypoints_abs = [(x, y, conf) for x, y, conf in keypoints]
            # Draw keypoints and skeleton
            image = image_1 if i == 0 else image_2
            draw = ImageDraw.Draw(image)#.format(i=i))
            radius = 5
            keypoint_color = (255, 0, 0)  # red
            skeleton_color = (0, 255, 0)  # green

            # Draw keypoints
            for x, y, conf in keypoints_abs:
                if conf > 0.2:  # draw only if confidence is higher than 0.5
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=keypoint_color)

            # Draw skeleton
            for start, end in skeleton:
                start_keypoint = keypoints_abs[start - 1]
                end_keypoint = keypoints_abs[end - 1]
                #print(start_keypoint[2])
                if start_keypoint[2] > 0.2 and end_keypoint[2] > 0.2:  # draw only if both keypoints have confidence higher than 0.5
                    draw.line([start_keypoint[0], start_keypoint[1], end_keypoint[0], end_keypoint[1]], fill=skeleton_color, width=2)
                    
    # Show the image
    ax[0].imshow(image_1)
    ax[1].imshow(image_2)
    plt.show()

def output_to_json(pred, img_name):
    bbox = pred[:4].detach().cpu().numpy()
    keypoints = pred[6:].detach().cpu().numpy()
    #print(len(keypoints))
    keypoints = list(zip(
        keypoints[0::3],
        keypoints[1::3],
        keypoints[2::3],
    ))
    
    #print(keypoints)
    json_entry = {
    'image_id' : img_name,
    'category_id' : 1,
    'keypoints' : keypoints,
    'score' : pred[4].detach().cpu().numpy(),
    'bbox' : bbox
    }
    return json_entry
    #print(40*'=')
    #print(json_entry)
    #print(40*'=')

def predict_coco():
    
    setup = 'local'
    if setup == 'local':    
        cocoGt = pycocotools.coco.COCO(os.path.join(os.path.expanduser('~'),'../../media/dim/linux_drive/datasets/coco/annotations/person_keypoints_val2017.json'))
        images_dir = os.path.join(os.path.expanduser('~'),'../../media/dim/linux_drive/datasets/coco/val2017')
    if setup == 'hpc':
        cocoGt = pycocotools.coco.COCO(os.path.join(os.path.expanduser('~'),'../../../../work3/dimara/data/data-mscoco/annotations/person_keypoints_val2017.json'))
        images_dir = os.path.join(os.path.expanduser('~'),'../../../../work3/dimara/data/data-mscoco/images/val2017')

    catIds = cocoGt.getCatIds('person')
    imgIds = cocoGt.getImgIds(catIds=catIds)
    imgIds = imgIds#[:32]
    
    torch_yolo_pose = torch.load('yolov8s-pose.pt')
    multimodel = torch_yolo_pose['model'].to('cuda:0')
    
    json_anns_yolo = []
    LOG.info('Predicting on {} images of COCO keypoints val2017 dataset'.format(len(imgIds)))
    for n, imgId in enumerate(tqdm(imgIds)):
        
        # read image
        img = cocoGt.imgs[imgId]
        img_path = os.path.join(images_dir, img['file_name'])
        image = Image.open(img_path).convert('RGB')
        width_height = image.size
        shape = torch.zeros(1,1,width_height[0],width_height[1])
        processed_image = transform(image).to('cuda:0').half()
        # Forward pass and postprocess predictions
        output = multimodel(processed_image.unsqueeze(0))
        final_output = postprocess(output[0]) 
           
        for i in range(len(final_output[0])):    

            keypoints = final_output[0][i][6:].detach().cpu().numpy()
            keypoints = list(zip(
                keypoints[0::3],
                keypoints[1::3],
                keypoints[2::3],
            ))
            resized_keypoints,flat_keypoints,score = [],[],0
            
            for keypoint in keypoints:
                score += keypoint[2]
                resized_keypoints.append((keypoint[0] * (width_height[0] / 640), keypoint[1] * (width_height[1]/640), keypoint[2]))
            score = score / len(keypoints)
            for sublist in resized_keypoints:
                flat_keypoints.extend(sublist)

            float64_keypoints = np.array(flat_keypoints).astype(np.float64).tolist()
            #print(float64_keypoints)
            #print(keypoints)
            json_entry = {
            'image_id' : imgId,
            'category_id' : 1,
            'keypoints' : float64_keypoints,
            'score' : score#pred[0][4].detach().cpu().numpy().astype(np.float64),
            #'bbox' : bbox
            }
            
                    
            json_anns_yolo.append(json_entry)

    with open('results_mine.json', 'w') as f:
        json.dump(json_anns_yolo, f)
    #print(f'one_pred: {len(one_pred_list)}, two_pred: {len(two_pred_list)}, more_pred: {len(more_pred_list)}')
    #print(f'one_pred: {one_pred_list}, two_pred: {two_pred_list}, more_pred: {more_pred_list}')
    cocoDt = cocoGt.loadRes('results_mine.json')
    #print(imgIds)
    
    LOG.info('Evalauting on COCO dataset'.format(len(imgIds)))
    cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2,eps = 0.001, momentum = 0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        #print(x.shape)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class MultiScaleFeatureFusionTransformer(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureFusionTransformer, self).__init__()
        
        # Define query, key, and value projection layers
        self.query_conv = nn.Conv2d(in_channels, in_channels//64, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_channels, in_channels//64, kernel_size=1, stride=1, padding=0)
        self.value_conv = nn.Conv2d(in_channels, 1120, kernel_size=1, stride=1, padding=0)
        # Define positional encoding layer
        self.positional_encoding = nn.Parameter(torch.zeros(1, in_channels, 80, 80))

        # Define transformer layer
        self.transformer = nn.Transformer(d_model=1120, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        # Define output projection layer
        self.output = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        # Concatenate features across resolution levels
        x1, x2, x3 = inputs
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3], dim=1)
        #x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=True)

        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Reshape query and key to match the input requirements of nn.Transformer
        query = query.permute(3, 2, 0, 1).contiguous().view(x.size(-1), -1, x.size(0))
        key = key.permute(3, 2, 0, 1).contiguous().view(x.size(-1), -1, x.size(0))

        # Add positional encoding to the input
        x = x + self.positional_encoding

        # Compute self-attention using the transformer
        x = x.permute(3, 2, 0, 1)
        value = self.transformer(query, key, value)
        x = value.permute(2, 3, 1, 0)

        # Project features to output
        x = self.output1(x)
        x = F.sigmoid(x)
        x = self.output2(x)

        return x

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels=[512, 256, 128], out_channels=32):
        super(MultiScaleFeatureFusion, self).__init__()

        # Define query, key, and value projection layers for each input resolution level
        self.query_convs = nn.ModuleList([nn.Conv2d(c, c // 8, kernel_size=1) for c in in_channels])
        self.key_convs = nn.ModuleList([nn.Conv2d(c, c // 8, kernel_size=1) for c in in_channels])
        self.value_convs = nn.ModuleList([nn.Conv2d(c, c, kernel_size=1) for c in in_channels])

        # Define attention layers for each input resolution level
        self.attention_convs = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=1) for c in in_channels])

        # Define output projection layers
        self.output_conv1 = nn.Conv2d(sum(in_channels), out_channels, kernel_size=1)
        self.output_conv2 = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, input):
        # Resize x2 and x1 to match the resolution of x3
        x3,x2,x1 = input
        x2 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=x3.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate features across resolution levels
        x =[x1, x2, x3]

        # Compute attention weights for each resolution level
        attention_weights = []
        for i in range(len(self.query_convs)):
            query = self.query_convs[i](x[i])
            key = self.key_convs[i](x[i])
            value = self.value_convs[i](x[i])

            # Compute self-attention
            query = query.view(query.size(0), -1, query.size(2) * query.size(3))
            key = key.view(key.size(0), -1, key.size(2) * key.size(3))
            value = value.view(value.size(0), -1, value.size(2) * value.size(3))

            attention = torch.bmm(query.permute(0, 2, 1), key)
            attention = F.softmax(attention, dim=-1)

            x_att = torch.bmm(value, attention.permute(0, 2, 1))
            x_att = x_att.view(value.size(0), -1, *x[i].size()[2:])
            x_att = self.attention_convs[i](x_att)

            attention_weights.append(x_att)

        # Combine features using attention weights
        x = torch.cat([x3, x2, x1], dim=1)
        x_att = torch.cat(attention_weights, dim=1)
        # Multiply attention maps elementwise
        x_att_prod = x_att[:, 0, :, :] * x_att[:, 1, :, :] * x_att[:, 2, :, :]

        # Reshape x_att_prod to match x
        #x_att_prod = x_att_prod.unsqueeze(1).repeat(1, 1280, 1, 1)
        x_att_prod = x_att_prod.unsqueeze(1).repeat(1, 896, 1, 1)

        # Multiply x with attention map
        x = x * x_att_prod

        # Project features to output
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.output_conv2(x)
        print(torch.min(x))
        print(torch.max(x))
        print(torch.median(x))
        #print(x.shape)

        return x

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        #print(f'c2fshape{ x.shape}')
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        print(f'c2fshape{ x.shape}')

        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        #print(type(x))
        return torch.cat(x, self.d)

class RebuiltYOLO(nn.Module):
    def __init__(self,task = 'seg'):
        super(RebuiltYOLO, self).__init__()
        
        self.add_module('0' , Conv(3, 32, 3, 2))
        self.add_module('1' , Conv(32, 64, 3, 2))
        self.add_module('2' , C2f(64, 64, 1, shortcut=True, g=1, e=0.5))
        self.add_module('3' , Conv(64, 128, 3, 2))
        self.add_module('4' , C2f(128, 128, 2, shortcut=True, g=1, e=0.5))
        self.add_module('5' , Conv(128, 256, 3, 2))
        self.add_module('6' , C2f(256, 256, 2, shortcut=True, g=1, e=0.5))
        self.add_module('7' , Conv(256, 512, 3, 2))
        self.add_module('8' , C2f(512, 512, 1, shortcut=True, g=1, e=0.5))
        self.add_module('9' , SPPF(512, 512, k=5))
        self.add_module('10', nn.Upsample(scale_factor=2.0, mode='nearest'))
        self.add_module('11', Concat(1))
        self.add_module('12', C2f(768, 256, 1, shortcut=False, g=1, e=0.5))
        self.add_module('13', nn.Upsample(scale_factor=2.0, mode='nearest'))
        self.add_module('14', Concat(1))
        self.add_module('15', C2f(384, 128, 1, shortcut=False, g=1, e=0.5))
        self.add_module('16', Conv(128, 128, 3, 2))
        self.add_module('17', Concat(1))
        self.add_module('18', C2f(384, 256, 1, shortcut=False, g=1, e=0.5))
        self.add_module('19', Conv(256, 256, 3, 2))
        self.add_module('20', Concat(1))
        self.add_module('21', C2f(768, 512, 1, shortcut=False, g=1, e=0.5))
        if task == 'pose':
            self.add_module('22', Pose(nc=1,ch=(128, 256, 512)))

        elif task == 'seg':
            self.add_module('22', Segment(nc=80, nm=32, npr =128, ch=(128, 256, 512)))
            self.add_module('23', MultiScaleFeatureFusion(in_channels=[512, 256, 128], out_channels=32))
            #self.add_module('23', Conv(80, 2, 3, 2))

            
    def forward(self, x):
        x_orig = x
        x = self._modules['0'](x)
        x = self._modules['1'](x)
        x = self._modules['2'](x)
        x = self._modules['3'](x)
        x_4 = self._modules['4'](x)
        x = self._modules['5'](x_4)
        x_6 = self._modules['6'](x)
        x = self._modules['7'](x_6)
        x = self._modules['8'](x)
        x_9 = self._modules['9'](x)
        x = self._modules['10'](x_9)
        x = self._modules['11']([x, x_6])
        x_12 = self._modules['12'](x)
        x = self._modules['13'](x_12)
        x = self._modules['14']([x, x_4])
        x_15 = self._modules['15'](x)
        x = self._modules['16'](x_15) 
        x = self._modules['17']([x, x_12])
        x_18 = self._modules['18'](x)
        x = self._modules['19'](x_18)
        x = self._modules['20']([x, x_9])
        x_21 = self._modules['21'](x)
        #print(x_15.shape,x_18.shape,x_21.shape)

        #if task == 'pose':
        #    x = self._modules['22']([x_15,x_18,x_21])
        #if task == 'seg':
        x = self._modules['22']([x_15,x_18,x_21])
        #print(x)
        y = self._modules['23']([x_15,x_18,x_21])
        #x = 
        
        y = nn.functional.interpolate(y, size=x_orig.shape[2:], mode='bilinear', align_corners=False)
        y = nn.functional.sigmoid(y)
        return x,y
    
    
    

class RebuiltYOLOLarge(nn.Module):
    def __init__(self,task = 'seg'):
        super(RebuiltYOLOLarge, self).__init__()
        
        self.add_module('0' , Conv(3, 64, 3, 2))
        self.add_module('1' , Conv(64, 128, 3, 2))
        self.add_module('2' , C2f(128, 128, 3, shortcut=True, g=1, e=0.5))
        self.add_module('3' , Conv(128, 256, 3, 2))
        self.add_module('4' , C2f(256, 256, 6, shortcut=True, g=1, e=0.5))
        self.add_module('5' , Conv(256, 512, 3, 2))
        self.add_module('6' , C2f(512, 512, 6, shortcut=True, g=1, e=0.5))
        self.add_module('7' , Conv(512, 512, 3, 2))
        self.add_module('8' , C2f(512, 512, 3, shortcut=True, g=1, e=0.5))
        self.add_module('9' , SPPF(512, 512, k=5))
        self.add_module('10', nn.Upsample(scale_factor=2.0, mode='nearest'))
        self.add_module('11', Concat(1))
        self.add_module('12', C2f(1024, 512, 3, shortcut=False, g=1, e=0.5))
        self.add_module('13', nn.Upsample(scale_factor=2.0, mode='nearest'))
        self.add_module('14', Concat(1))
        self.add_module('15', C2f(768, 256, 3, shortcut=False, g=1, e=0.5))
        self.add_module('16', Conv(256, 256, 3, 2))
        self.add_module('17', Concat(1))
        self.add_module('18', C2f(768, 512, 3, shortcut=False, g=1, e=0.5))
        self.add_module('19', Conv(512, 512, 3, 2))
        self.add_module('20', Concat(1))
        self.add_module('21', C2f(1024, 512, 3, shortcut=False, g=1, e=0.5))
        if task == 'pose':
            self.add_module('22', Pose(nc=1,ch=(256, 512, 1024)))

        elif task == 'seg':
            #self.add_module('22', Segment(nc=2, nm=32, npr =256, ch=(256, 512, 512)))
            self.add_module('23', MultiScaleFeatureFusion(in_channels=[512,512, 256], out_channels=64))
            #self.add_module('23', MultiScaleFeatureFusion(in_channels=[512, 256, 128], out_channels=32))


            
    def forward(self, x):
        x_orig = x
        x = self._modules['0'](x)
        x = self._modules['1'](x)
        x = self._modules['2'](x)
        x = self._modules['3'](x)
        x_4 = self._modules['4'](x)
        x = self._modules['5'](x_4)
        x_6 = self._modules['6'](x)
        x = self._modules['7'](x_6)
        x = self._modules['8'](x)
        x_9 = self._modules['9'](x)
        x = self._modules['10'](x_9)
        x = self._modules['11']([x, x_6])
        x_12 = self._modules['12'](x)
        x = self._modules['13'](x_12)
        x = self._modules['14']([x, x_4])
        x_15 = self._modules['15'](x)
        x = self._modules['16'](x_15) 
        x = self._modules['17']([x, x_12])
        x_18 = self._modules['18'](x)
        x = self._modules['19'](x_18)
        x = self._modules['20']([x, x_9])
        x_21 = self._modules['21'](x)
        #print(x_15.shape,x_18.shape,x_21.shape)
        #if task == 'pose':
        #    x = self._modules['22']([x_15,x_18,x_21])
        #if task == 'seg':
        #x = self._modules['22']([x_15,x_18,x_21])
        y = self._modules['23']([x_15,x_18,x_21])
        y = nn.functional.interpolate(y, size=x_orig.shape[2:], mode='bilinear', align_corners=False)
        y = nn.functional.sigmoid(y)
        return x,y
    
'''
LOG = logging.getLogger(__name__)

# Define transformation
transform = transforms.transforms.Compose([transforms.ToTensor(), transforms.Resize((640,640))])

tasks = ['pose','seg','multi']
task = tasks[1]

# LOAD POSE NETWORK
if task == 'pose':
    #torch_yolo_pose = torch.load('yolov8s-pose.pt')
    #torch_model = torch_yolo_pose['model'].to('cuda:0')
    saved_state_dict = torch.load('yolov8s-pose.pt')['model'].state_dict()
   
elif task == 'seg':
    #torch_segmentation = torch.load('yolov8s-seg.pt')
    #torch_model = torch_segmentation['model'].to('cuda:0')
    saved_state_dict = torch.load('yolov8s-seg.pt')['model'].state_dict()
    
updated_state_dict = {}
for layer_name, weights in saved_state_dict.items():
    updated_layer_name = '.'.join(layer_name.split('.')[1:])
    updated_state_dict[updated_layer_name] = weights

multimodel = RebuiltYOLO(task).to('cuda:0').eval().half()
multimodel.load_state_dict(updated_state_dict, strict=False)

#seg_model = torch.load('yolov8l-seg.pt')['model'].to('cuda:0')

#PREDICTION_OPTIONS
#predict_coco()
#yolo_pose_example()

#SEGMENTATION TASK
# Load image
img_path_1 = 'test/test_24a_2.png'
image_1 = Image.open(img_path_1).convert('RGB')#.resize(IMAGE_SHAPE)
processed_image_1 = transform(image_1).to('cuda:0').half()
width_height = image_1.size
image_1 = image_1.resize((640,640))
shape_tensor = torch.zeros([1, 3, width_height[0], width_height[1]], device='cuda:0').half()
#print(processed_image_1.shape)

with torch.no_grad():
    # Forward pass and postprocess predictions
    output_seg, output_floor = multimodel(processed_image_1.unsqueeze(0))
    final_output =  seg_postprocess(output_seg,processed_image_1.unsqueeze(0),image_1,img_path_1)
    print(output_floor.shape)
    #final_output = postprocess(output[0])
    #print(final_output)
    #res_plotted = final_output[0].plot()
    #res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
    #res_plotted = cv2.resize(res_plotted, (width_height[0],width_height[1]))
    #cv2.imshow("result", res_plotted)
    #cv2.waitKey(0)
'''