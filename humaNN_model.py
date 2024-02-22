# pylint: disable=too-many-statements, too-many-branches

"""
Loco super class for MonStereo, MonoLoco, MonoLoco++ nets.
From 2D joints to real-world distances with monocular &/or stereo cameras
"""

import os
import math
import logging
from collections import defaultdict
import random

import numpy as np
import torch

from PIL import Image
from monoloco.utils import get_iou_matches, reorder_matches, get_keypoints, pixel_to_camera, xyz_from_distance,  \
    mask_joint_disparity
from monoloco.network.process import preprocess_monstereo, preprocess_monoloco, extract_outputs, extract_outputs_mono,\
    filter_outputs, cluster_outputs, unnormalize_bi, laplace_sampling
from monoloco.activity import social_interactions, is_raising_hand
from monoloco.monoloco_model import MonolocoModel, MonolocoModelSparse,MonolocoModelhumaNN,MonolocoModelhumaNNMoD, EarlyGuidance
from monoloco.humann_model import ModifiedHumanModel_2, TransformerHumaNN#,ModifiedHumanModel_2,ModifiedHumanModel_3,ModifiedHumanModel_4,ModifiedHumanModel_5
from monoloco.humann_model import TestModel_3s,TestModel_3m,TestModel_3d,TestModel_3s_d, Lazaros, IdentityMapping,ModifiedLinearBlock,SecondStage

from torchview import draw_graph
import graphviz
graphviz.set_jupyter_format('png')

class Loco:
    """Class for both MonoLoco and MonStereo"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    LINEAR_SIZE_MONO = 256
    N_SAMPLES = 100

    def __init__(self, model, mode, net=None, device=None, n_dropout=0, p_dropout=0.2, linear_size=256, num_stage = 3):

        # Select networks
        assert mode in ('mono', 'stereo'), "mode not recognized"
        self.mode = mode
        self.model = model
        self.net = net
        input_size = 34
        output_size = 2
        aux_input_size = 51
        self.num_stage = num_stage
        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.n_dropout = n_dropout
        self.epistemic = bool(self.n_dropout > 0)

        # if the path is provided load the model parameters
        for i in range(2):
            print(i)
            if isinstance(model, str):
                model_path = model
                if net in ('monoloco', 'monoloco_p'):
                    self.model = MonolocoModel(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                            output_size=output_size, num_stage=self.num_stage)
                    self.auxmodel = None
                    #self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage),strict=False)
                elif net == 'monoloco_pretrained':
                    self.model = MonolocoModelSparse(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size)
                    self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage),strict=False)
                elif net == 'humann':
                    self.model = MonolocoModelhumaNN(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                            output_size=output_size)         
                elif net == 'humann_modif':
                    self.model = MonolocoModelhumaNNMoD(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size)
                #elif net == 'early_guidance':
                #    self.model = EarlyGuidance(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                #                output_size=output_size)
                elif net == 'transformer':
                    self.model = TransformerHumaNN(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size,num_stage=self.num_stage)
                    self.auxmodel = None

                elif net == 'transformer_3d':
                    self.model = ModifiedHumanModel_2(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)
                elif net == 'test_3s':
                    self.model = TestModel_3s(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)    
                elif net == 'test_3m':
                    self.model = TestModel_3m(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)
                elif net == 'test_3d':
                    self.model = TestModel_3d(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)    
                
                elif net == 'test_3s_d':
                    self.model = TestModel_3s_d(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)     
                    
                #elif net == 'modified_1':
                #    self.model = ModifiedHumanModel_1(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                #                output_size=output_size, num_stage=self.num_stage)  
                elif net == 'lazaros':
                    self.model = ModifiedHumanModel_1(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)  
                    #self.auxmodel = Lazaros(input_size=17, linear_size=256)
                    #self.auxmodel = ModifiedHumanModel_1(p_dropout=p_dropout, input_size=17, linear_size=linear_size,
                    #            output_size=output_size, num_stage=self.num_stage) 
                    self.auxmodel.eval()
                    self.auxmodel.to(self.device)
                    
                elif net == 'identity':
                    self.model = MonolocoModel(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)  
                    self.auxmodel = IdentityMapping()
                    #self.auxmodel = ModifiedHumanModel_1(p_dropout=p_dropout, input_size=17, linear_size=linear_size,
                    #            output_size=output_size, num_stage=self.num_stage) 
                    self.auxmodel.eval()
                    self.auxmodel.to(self.device)
                    
                elif net == 'two_stage':
                    self.model = MonolocoModel(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)  
                    self.auxmodel = SecondStage(aux_input_size = aux_input_size, linear_size=256, p_dropout=p_dropout)
                    
                    
                    #self.auxmodel =
                    #  ModifiedHumanModel_1(p_dropout=p_dropout, input_
                    #IdentityMapping
                    #self.auxmodel = IdentityMapping()#size=17, linear_size=linear_size,
                    #            output_size=output_size, num_stage=self.num_stage) 
                    self.auxmodel.eval()
                    self.auxmodel.to(self.device)
                elif net == 'two_stage_tran':
                    self.model = TransformerHumaNN(p_dropout=p_dropout, input_size=input_size, linear_size=linear_size,
                                output_size=output_size, num_stage=self.num_stage)  
                    self.auxmodel = SecondStage(linear_size=256, p_dropout=p_dropout)
                    self.auxmodel.eval()
                    self.auxmodel.to(self.device)
                '''
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.w2_modified.parameters():
                    param.requires_grad = True
                for param in self.model.pre_w2.parameters():
                    param.requires_grad = True
                for param in self.model.batch_norm2.parameters():
                    param.requires_grad = True
                for param in self.model.relu2.parameters():
                    param.requires_grad = True
                for param in self.model.dropout2.parameters():
                    param.requires_grad = True
                '''

            else:
                self.model = model
                
            if i == 0:
                #check if file exists
                if os.path.isfile(f'model.gv.png'):
                    #delete file
                    os.remove(f'model.gv.png')
                #visualize_model = self.model
                model_graph = draw_graph(self.model, input_size=(1, input_size), device='meta',save_graph=True,directory='.')
                #rename file
                os.rename(f'model.gv.png',f'model_{net}.gv.png')
                if self.auxmodel:
                    model_graph = draw_graph(self.auxmodel, input_size=(1, aux_input_size), device='meta',save_graph=True,directory='.')
                    os.rename(f'model.gv.png',f'auxmodel_{net}.gv.png')
                else:
                    #create black png file
                    img = Image.new('RGB', (20, 100), color = (0, 0, 0))
                    img.save(f'auxmodel_{net}.gv.png')
                print('saved model graph')

                    
                
                #model_graph.visual_graph
            else:
                self.model.eval()  # Default is train
                self.model.to(self.device)


    def forward(self, keypoints, kk, bbox, keypoints_r=None, status = 'test'):
        """
        Forward pass of MonSter or monoloco network
        It includes preprocessing and postprocessing of data
        """
        #if status == 'train':
        #    self.model.train()
        #    self.auxmodel.train()
        #if not keypoints:
        #    return None

        #print(keypoints.shape)
        keypoints = keypoints.clone().detach().to(self.device)
        #keypoints = torch.tensor(keypoints).to(self.device)

        tensor_keypoints = keypoints[:,:3,:]
        distance_keypoints = keypoints[:,3,:].type(torch.FloatTensor).to(self.device)
        #print(keypoints)
        #new_keypoints = torch.cat([keypoints] * 2, dim=0)  # Concatenate along the batch dimension (first dimension)

        #print(keypoints)
        #keypoints = keypoints.permute(0, 2, 1).double()
        kk = torch.tensor(kk).to(self.device)
        #new_kk = torch.cat([kk] * 2, dim=0)  # Concatenate along the batch dimension (first dimension)

        #print(40*'*')
        #print(40*'*')
        #print(f'keypoints before {tensor_keypoints}')
        inputs = preprocess_monoloco(tensor_keypoints, kk, zero_center=True)
        #print(inputs.dtype)
        #rint(tensor_keypoints.dtype)
        #print(distance_keypoints.dtype)
        #print(40*'*')
        
        #print(inputs.shape)
        #print(distance_keypoints.shape)
        #print(f'keypoints afta {inputs}')
        #print(40*'*')
        #print(40*'*')
        combined_tensor = torch.cat((inputs, distance_keypoints), dim=1)
        #print(f'shape combi ned tensor {combined_tensor.shape}')
        outputs = self.model(combined_tensor)
        bi = unnormalize_bi(outputs)
        predicted_depth  = outputs[:, 0:1]
        predicted_depth_s = predicted_depth.flatten()
        if len(predicted_depth_s) > 1:
            print(f'original pred {predicted_depth_s}')

        #if self.net == 'monoloco':
            
        if self.net in ['identity','two_stage','two_stage_tran']:
            #Freeze auxmodel
            distance_keypoints = distance_keypoints / 100
            #print(f'predicted {predicted_depth}')
            #print(f'distancekeypoints {distance_keypoints}')
        
            #print(f'predicted_depth {predicted_depth}')
            # Reshaping tensor_b to match the shape of tensor_a
            tensor_b_reshaped = predicted_depth_s.view(-1, 1)

            # Adding the corresponding value from tensor_b to the zeros in tensor_a
            updated_tensor = distance_keypoints + (distance_keypoints == 0) * tensor_b_reshaped
            updated_tensor_test = torch.zeros_like(updated_tensor)
            updated_tensor_test = updated_tensor_test + (updated_tensor_test == 0) * tensor_b_reshaped
            #Average values of updated tensor distance_keypoints withou zero values
            average_tensor = torch.mean(updated_tensor[updated_tensor != 0])
            #assign average value into one of random 17 positions in updated tensor test
            i = random.randint(0, 16)
            
            #updated_tensor_test[0][i] = average_tensor
            updated_tensor_test = updated_tensor_test + (updated_tensor_test == 0) * average_tensor
            #print(updated_tensor.shape)
            # count non zero values in updated tensor
            #if updated_tensor[updated_tensor != 0].shape[0] > 0:
                #print(f'updated tensor {updated_tensor[updated_tensor != 0]}')
                #print(updated_tensor)
                #print(updated_tensor_test)
            #empty_tensor = torch.ones_like(updated_tensor)
            updated_tensor = torch.cat((inputs, updated_tensor), dim=1)

            new_pred = self.auxmodel(updated_tensor)
            new_pred  = new_pred[:, 0:1]
            #print(f'new_pred multiple values {new_pred}')
            
            #updated_tensor = torch.cat((inputs, updated_tensor_test), dim=1)

            #new_pred = self.auxmodel(updated_tensor)
            #new_pred  = new_pred[:, 0:1]
            #print(f'new_pred one only values {new_pred}')


            #print(predicted_depth.shape)
            #print(new_pred)
            if len(new_pred.flatten()) > 1:
            
                print(f'auxiliary_pred {new_pred.flatten()}')
            predicted_depth = new_pred
            #predicted_depth.flatten())

        dic_out = {'d': predicted_depth, 'bi': bi}

        

        if self.n_dropout > 0 and self.net != 'monstereo':
            varss = self.epistemic_uncertainty(inputs)
            dic_out['epi'] = varss
        else:
            dic_out['epi'] = [0.] * outputs.shape[0]
            # Add in the dictionary

        return dic_out

    def epistemic_uncertainty(self, inputs):
        """
        Apply dropout at test time to obtain combined aleatoric + epistemic uncertainty
        """
        assert self.net in ('monoloco', 'monoloco_p', 'monoloco_pp','humann'), "Not supported for MonStereo"

        self.model.dropout.training = True  # Manually reactivate dropout in eval
        total_outputs = torch.empty((0, inputs.size()[0])).to(self.device)

        for _ in range(self.n_dropout):
            outputs = self.model(inputs)

            # Extract localization output
            if self.net in ('monoloco','humann'):
                db = outputs[:, 0:2]
            else:
                db = outputs[:, 2:4]

            # Unnormalize b and concatenate
            bi = unnormalize_bi(db)
            outputs = torch.cat((db[:, 0:1], bi), dim=1)

            samples = laplace_sampling(outputs, self.N_SAMPLES)
            total_outputs = torch.cat((total_outputs, samples), 0)
        varss = total_outputs.std(0)
        self.model.dropout.training = False
        return varss

    @staticmethod
    def post_process(dic_in, boxes, keypoints, kk, dic_gt=None, iou_min=0.3, reorder=True, verbose=False):
        """Post process monoloco to output final dictionary with all information for visualizations"""

        dic_out = defaultdict(list)
        if dic_in is None:
            return dic_out

        if dic_gt:
            boxes_gt = dic_gt['boxes']
            dds_gt = [el[3] for el in dic_gt['ys']]
            matches = get_iou_matches(boxes, boxes_gt, iou_min=iou_min)
            dic_out['gt'] = [True]
            if verbose:
                print("found {} matches with ground-truth".format(len(matches)))

            # Keep track of instances non-matched
            idxs_matches = [el[0] for el in matches]
            not_matches = [idx for idx, _ in enumerate(boxes) if idx not in idxs_matches]

        else:
            matches = []
            not_matches = list(range(len(boxes)))
            if verbose:
                print("NO ground-truth associated")

        if reorder and matches:
            matches = reorder_matches(matches, boxes, mode='left_right')

        all_idxs = [idx for idx, _ in matches] + not_matches
        dic_out['gt'] = [True]*len(matches) + [False]*len(not_matches)

        uv_shoulders = get_keypoints(keypoints, mode='shoulder')
        uv_heads = get_keypoints(keypoints, mode='head')
        uv_centers = get_keypoints(keypoints, mode='center')
        xy_centers = pixel_to_camera(uv_centers, kk, 1)

        # Add all the predicted annotations, starting with the ones that match a ground-truth
        for idx in all_idxs:
            kps = keypoints[idx]
            box = boxes[idx]
            dd_pred = float(dic_in['d'][idx])
            bi = float(dic_in['bi'][idx])
            var_y = float(dic_in['epi'][idx])
            uu_s, vv_s = uv_shoulders.tolist()[idx][0:2]
            uu_c, vv_c = uv_centers.tolist()[idx][0:2]
            uu_h, vv_h = uv_heads.tolist()[idx][0:2]
            uv_shoulder = [round(uu_s), round(vv_s)]
            uv_center = [round(uu_c), round(vv_c)]
            uv_head = [round(uu_h), round(vv_h)]
            xyz_pred = xyz_from_distance(dd_pred, xy_centers[idx])[0]
            distance = math.sqrt(float(xyz_pred[0])**2 + float(xyz_pred[1])**2 + float(xyz_pred[2])**2)
            conf = 0.035 * (box[-1]) / (bi / distance)

            dic_out['boxes'].append(box)
            dic_out['confs'].append(conf)
            dic_out['dds_pred'].append(dd_pred)
            dic_out['stds_ale'].append(bi)
            dic_out['stds_epi'].append(var_y)

            dic_out['xyz_pred'].append(xyz_pred.squeeze().tolist())
            dic_out['uv_kps'].append(kps)
            dic_out['uv_centers'].append(uv_center)
            dic_out['uv_shoulders'].append(uv_shoulder)
            dic_out['uv_heads'].append(uv_head)

            # For MonStereo / MonoLoco++
            try:
                dic_out['angles'].append(float(dic_in['yaw'][0][idx]))  # Predicted angle
                dic_out['angles_egocentric'].append(float(dic_in['yaw'][1][idx]))  # Egocentric angle
            except KeyError:
                continue

            # Only for MonStereo
            try:
                dic_out['aux'].append(float(dic_in['aux'][idx]))
            except KeyError:
                continue

        for idx, idx_gt in matches:
            dd_real = dds_gt[idx_gt]
            xyz_real = xyz_from_distance(dd_real, xy_centers[idx])
            dic_out['dds_real'].append(dd_real)
            dic_out['boxes_gt'].append(boxes_gt[idx_gt])
            dic_out['xyz_real'].append(xyz_real.squeeze().tolist())
        return dic_out

    @staticmethod
    def social_distance(dic_out, args):

        angles = dic_out['angles']
        dds = dic_out['dds_pred']
        stds = dic_out['stds_ale']
        xz_centers = [[xx[0], xx[2]] for xx in dic_out['xyz_pred']]

        # Prepare color for social distancing
        dic_out['social_distance'] = [bool(social_interactions(idx, xz_centers, angles, dds,
                                                               stds=stds,
                                                               threshold_prob=args.threshold_prob,
                                                               threshold_dist=args.threshold_dist,
                                                               radii=args.radii))
                                      for idx, _ in enumerate(dic_out['xyz_pred'])]
        return dic_out


    @staticmethod
    def raising_hand(dic_out, keypoints):
        dic_out['raising_hand'] = [is_raising_hand(keypoint) for keypoint in keypoints]
        return dic_out


def median_disparity(dic_out, keypoints, keypoints_r, mask):
    """
    Ablation study: whenever a matching is found, compute depth by median disparity instead of using MonSter
    Filters are applied to masks nan joints and remove outlier disparities with iqr
    The mask input is used to filter the all-vs-all approach
    """

    keypoints = keypoints.cpu().numpy()
    keypoints_r = keypoints_r.cpu().numpy()
    mask = mask.cpu().numpy()
    avg_disparities, _, _ = mask_joint_disparity(keypoints, keypoints_r)
    BF = 0.54 * 721
    for idx, aux in enumerate(dic_out['aux']):
        if aux > 0.5:
            idx_r = np.argmax(mask[idx])
            z = BF / avg_disparities[idx][idx_r]
            if 1 < z < 80:
                dic_out['xyzd'][idx][2] = z
                dic_out['xyzd'][idx][3] = torch.norm(dic_out['xyzd'][idx][0:3])
    return dic_out
