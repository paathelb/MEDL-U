from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from datasets.kitti_detection import KittiDetectionDataset
from datasets.nuscenes_detection import NuscenesDetectionDataset
from datasets.kitti_loader import build_kitti_loader, move_to_cuda, merge_two_batch, make_tensor_keys
#from datasets.nuscenes_loader import build_nuscenes_loader, move_to_cuda, merge_two_batch, make_tensor_keys
import yaml
from easydict import EasyDict
import argparse
import torch
import os
from os import path, makedirs
from models.MTrans import MTrans
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import WarmupCosineAnnealing
import random
import numpy as np
from utils.stat_scores import HistoCounter, ScoreCounter
import pickle

from torch import nn
from loss import cal_diou_3d
import numpy as np
import math
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.data_classes import Box

import torch.distributed as dist
import torch 

import shutil

from method.run import runner

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

def freeze_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_checkpoint(save_path, epoch, model, optim=None, scheduler=None):
    if is_main_process():

    # do that ….

    # save, load models, download data etc….
        if not path.exists(path.dirname(save_path)):
            makedirs(path.dirname(save_path))
        print(f">>> Saving checkpoint as: {save_path}")
        model_state_dict = model.state_dict()
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model_state_dict
        }
        if optim is not None:
            ckpt['optimizer_state_dict'] = optim.state_dict()
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(ckpt, save_path)

def load_checkpoint(file_path, model, optim=None, scheduler=None):
    ckpt = torch.load(file_path)
    model_ckpt = ckpt['model_state_dict']
    model.load_state_dict(model_ckpt)
    if optim is not None:
        optim.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt['epoch']
    return epoch

def get_pbar_text(counter, prefix):
    stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf'])
    pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}"
    return pbar_text

def clamp_orientation_range(angles):
    # angles: (B, 1)
    a = angles.clone()          # Bl x 1
    for i in range(a.size(0)):  # Angle should fall between -np.pi and np.pi
        while a[i] > np.pi:
            a[i] = a[i] - np.pi * 2
        while a[i] <= -np.pi:
            a[i] = a[i] + np.pi*2
    assert (a<=np.pi).all() and (a>=-np.pi).all()
    return a

def format_kitti_labels(pred_dict, data_dict, with_score=True, nusc=False):
        data_dict = EasyDict(data_dict)

        location, dimension, yaw = pred_dict['location'], pred_dict['dimension'], pred_dict['yaw']
        location = location + pred_dict['subcloud_center'] + pred_dict['second_offset']
        direction = pred_dict['direction'].argmax(dim=-1)
        yaw = adjust_direction(yaw, direction)
        labels = []
        
        for i in range(int(pred_dict['batch_size'].item())):
            c = data_dict.calibs[i]
            x, y, z = location[i]
            l, w, h = dimension[i]

            a = yaw[i]

            a = -(a + np.pi/2)
            while a > np.pi:
                a = a - np.pi * 2
            while a <= -np.pi:
                a = a + np.pi*2
            a = round(a.item(), 2)
            
            z = z - h/2
            loc = torch.stack([x, y, z], dim=-1)
            loc = c.lidar_to_rect(loc.detach().cpu().unsqueeze(0).numpy())[0]
            loc = loc.round(2)
            dim = torch.stack([h, w, l], dim=-1).detach().cpu().numpy()
            dim = dim.round(2)
            x, y, z = loc
            alpha = a + math.atan2(z,x)+1.5*math.pi
            if alpha > math.pi:
                alpha = alpha - math.pi * 2
            elif alpha <= -math.pi:
                alpha = alpha + math.pi*2
            box_2d = ' '.join([f'{x:.2f}' for x in data_dict.boxes_2d[i].detach().cpu().numpy()])
            dim = ' '.join([f'{x:.2f}' for x in dim])
            loc = ' '.join([f'{x:.2f}' for x in loc])
            truncated = data_dict.truncated[i]
            occluded = data_dict.occluded[i]
            score = pred_dict['conf'][i].item()

            if nusc:
                score = pred_dict['conf'][i].item()
            else:
                if 'scores' in data_dict.keys():
                    # for test result, MAPGen confidence * 2D Box score
                    score = score * data_dict['scores'][i] / max(pred_dict['conf']).item()

            if with_score:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f} {score:.4f}')
            else:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f}')
        return labels, data_dict.frames

def train_one_epoch(cfg,
                    model, 
                    training_loader, 
                    unlabeled_training_loader,
                    optim, 
                    scheduler, 
                    counter, 
                    histo_counter,
                    epoch, 
                    writer,
                    rank):
    
    model.train()
    if unlabeled_training_loader is not None:
        process_bar = tqdm(training_loader, desc='E{epoch}')
        unlabeled_iter = iter(unlabeled_training_loader)
    else:
        process_bar = tqdm(training_loader, desc='E{epoch}')
    counter.reset()      
    histo_counter.reset()
     
    for data in process_bar:
        optim.zero_grad()
        #data = EasyDict(data)
        if unlabeled_training_loader is not None:
            try:
                unlabeled_data = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_training_loader)
                unlabeled_data = next(unlabeled_iter)
            #unlabeled_data = EasyDict(unlabeled_data)
            data = merge_two_batch(data, unlabeled_data)
        #data = make_tensor_keys(data)
        data = move_to_cuda(data, 'cuda', rank)
        
        pred_dict = model(data)

        if cfg.dist or cfg.is_dp:
            loss_dict, loss, iou3d_histo, loss_box= model.module.get_loss(pred_dict, data, rank)
        else:
            loss_dict, loss, iou3d_histo, loss_box = model.get_loss(pred_dict, data, rank)
        histo_counter.update(iou3d_histo)
        
        # statistics
        counter.update(loss_dict)
        #loss_box.register_hook(lambda grad: print(grad))
        loss.backward()
        # for p in model.parameters():
        #     print(p.grad.norm())
        
        optim.step()
        scheduler.step()
        counter.update({'lr':(optim.param_groups[0]['lr'], 1, 'learning_rate')})

        pbar_text = get_pbar_text(counter, f'T-{epoch}')        
        process_bar.set_description(pbar_text)
        
        # param_count = 0
        # param_nograd = 0
        # for param in model.parameters():
        #     param_count += 1
        #     try:
        #         if torch.count_nonzero(param.grad)==0:
        #             print(param)
        #             param_nograd += 1
        #             #print(f'{param} has no grad')
        #     except:
        #         print("Nonetype")
        #         print(param)
        #         param_nograd += 1

        # print(param_count)
        # print(param_nograd)
        
        # for n, p in model.named_parameters():
        #     
        #         print(f'{n} has no grad')

    stats = counter.average(None, group_by_description=True)
    for group in stats.keys():
        writer.add_scalars(f'Train/{group}', stats[group], epoch)
    writer.add_histogram('Train/iou_distribution', histo_counter.get_values(), epoch)

def adjust_direction(yaw, dir):
    # yaw: (B, 1), dir: (B, 1) - long
    yaw = clamp_orientation_range(yaw)
    for i in range(yaw.size(0)):
        # check direction
        if dir[i]==1 and not (yaw[i]>=-np.pi/2 and yaw[i]<np.pi/2):
                yaw[i] = yaw[i] + np.pi
        elif dir[i]==0 and (yaw[i]>=-np.pi/2 and yaw[i]<np.pi/2):
                yaw[i] = yaw[i] + np.pi
    return yaw

def get_annos_dict(labels, frame_id, pred_dict, nusc, id_to_token, id_to_lidar_path, boxes_lidar_nusc):
    # This only caters to batch_size of 1
    name, truncated, occluded, alpha, bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4, dimensions_1, dimensions_2, \
    dimensions_3, location_1, location_2, location_3, rotation_y, score = labels[0].split()
    name = np.array([name])
    truncated = np.array([float(truncated)], dtype=np.float32)
    occluded = np.array([occluded], dtype=np.float32)
    alpha = np.array([alpha], dtype=np.float32)
    bbox = np.array([[bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4]], dtype=np.float32)
    dimensions = np.array([[dimensions_3, dimensions_1, dimensions_2]], dtype=np.float32)
    location = np.array([[location_1, location_2, location_3]], dtype=np.float32)
    rotation_y = np.array([rotation_y], dtype=np.float32)
    score = np.array([score], dtype=np.float32)
    
    location_lidar = pred_dict['location'] + pred_dict['subcloud_center'] + pred_dict['second_offset']
    location_lidar = location_lidar.cpu().detach().numpy()
    x_lidar, y_lidar, z_lidar = location_lidar[0]       # For i=0
    dimension_lidar = pred_dict['dimension'].cpu().detach().numpy()
    l_lidar, w_lidar, h_lidar = dimension_lidar[0]      # For i=0
    z_lidar = z_lidar - h_lidar/2
    location_lidar = np.stack([x_lidar, y_lidar, z_lidar], axis=-1)
    location_lidar = location_lidar[None,...]
    
    yaw_lidar = pred_dict['yaw']
    direction = pred_dict['direction'].argmax(dim=-1)
    yaw_lidar = adjust_direction(yaw_lidar, direction).cpu().detach().numpy()
    # a = yaw_lidar[0]        # For i=0
    # a = -(a + np.pi/2)
    # while a > np.pi:
    #     a = a - np.pi * 2
    # while a <= -np.pi:
    #     a = a + np.pi*2
    # yaw_lidar = a[None,...]
    try: boxes_lidar = np.concatenate((location_lidar, dimension_lidar, yaw_lidar), axis=1)
    except: 
        print(location_lidar.shape)
        print(dimension_lidar.shape)
        print(yaw_lidar.shape)
        import pdb; pdb.set_trace() 

    if nusc:
        annos = {}
        annos['name'] = name
        annos['score'] = score
        annos['boxes_lidar'] = boxes_lidar_nusc
        annos['pred_labels'] = np.array([1], dtype=np.int64)
        annos['frame_id_kitti_ver'] = frame_id[0]
        annos['frame_id'] = id_to_lidar_path[frame_id[0]]
        annos['metadata'] = {'token': id_to_token[frame_id[0]]}
        return annos

    else:
        annos = {}
        annos['name'] = name
        annos['truncated'] = truncated 
        annos['occluded'] = occluded
        annos['alpha'] = alpha 
        annos['bbox'] = bbox
        annos['dimensions'] = dimensions
        annos['location'] = location
        annos['rotation_y'] = rotation_y
        annos['score'] = score                  # Does not matter for pcdet evaluation?
        annos['boxes_lidar'] = boxes_lidar      # Does not matter for pcdet evaluation?
        annos['frame_id'] = frame_id[0]
        return annos

def get_transforms(frames, lbl):
        """
        Returns transforms for the input token.
        :param token: KittiDB unique id.
        :param root: Base folder for all KITTI data.
        :return: {
            'velo_to_cam': {'R': <np.float: 3, 3>, 'T': <np.float: 3, 1>}. Lidar to camera transformation matrix.
            'r0_rect': <np.float: 3, 3>. Rectification matrix.
            'p_left': <np.float: 3, 4>. Projection matrix.
            'p_combined': <np.float: 4, 4>. Combined rectification and projection matrix.
        }. Returns the transformation matrices. For details refer to the KITTI devkit.
        """

        calib_filename = '/home/hpaat/my_exp/MTrans/data/nuscenes/' + lbl + '/calib/' + frames + '.txt'

        lines = [line.rstrip() for line in open(calib_filename)]
        velo_to_cam = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32)
        velo_to_cam.resize((3, 4))

        r0_rect = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32)
        r0_rect.resize((3, 3))
        p_left = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32)
        p_left.resize((3, 4))

        # Merge rectification and projection into one matrix.
        p_combined = np.eye(4)
        p_combined[:3, :3] = r0_rect
        p_combined = np.dot(p_left, p_combined)
        return {
            'velo_to_cam': {
                'R': velo_to_cam[:, :3],
                'T': velo_to_cam[:, 3]
            },
            'r0_rect': r0_rect,
            'p_left': p_left,
            'p_combined': p_combined,
        }

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def get_boxes_lidar_nuscenes_format(line, frames, lbl):
    # calib_filename = ''
    # Parse this line into box information.
    parsed_line = KittiDB.parse_label_line(line[0])

    # if parsed_line['name'] in {'DontCare', 'Misc'}:
    #     continue

    center = parsed_line['xyz_camera']
    wlh = parsed_line['wlh']
    yaw_camera = parsed_line['yaw_camera']
    name = parsed_line['name']
    score = parsed_line['score']

    # Optional: Filter classes.
    # if filter_classes is not None and name not in filter_classes:
    #     continue

    # The Box class coord system is oriented the same way as as KITTI LIDAR: x forward, y left, z up.
    # For orientation confer: http://www.cvlibs.net/datasets/kitti/setup.php.

    # 1: Create box in Box coordinate system with center at origin.
    # The second quaternion in yaw_box transforms the coordinate frame from the object frame
    # to KITTI camera frame. The equivalent cannot be naively done afterwards, as it's a rotation
    # around the local object coordinate frame, rather than the camera frame.
    quat_box = Quaternion(axis=(0, 1, 0), angle=yaw_camera) * Quaternion(axis=(1, 0, 0), angle=np.pi/2)
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)

    # 2: Translate: KITTI defines the box center as the bottom center of the vehicle. We use true center,
    # so we need to add half height in negative y direction, (since y points downwards), to adjust. The
    # center is already given in camera coord system.
    box.translate(center + np.array([0, -wlh[2] / 2, 0]))

    # 3: Transform to KITTI LIDAR coord system. First transform from rectified camera to camera, then
    # camera to KITTI lidar.
    # Get transforms for this sample
    transforms = get_transforms(frames[0], lbl)

    box.rotate(Quaternion(matrix=transforms['r0_rect']).inverse)
    box.translate(-transforms['velo_to_cam']['T'])
    box.rotate(Quaternion(matrix=transforms['velo_to_cam']['R']).inverse)

    # 4: Transform to nuScenes LIDAR coord system.

    # KITTI LIDAR has the x-axis pointing forward, but our LIDAR points to the right. So we need to apply a
    # 90 degree rotation around to yaw (z-axis) in order to align.
    # The quaternions will be used a lot of time. We store them as instance variables so that we don't have
    # to create a new one every single time.
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    box.rotate(kitti_to_nu_lidar)

    # Set score or NaN.
    box.score = score

    # Set dummy velocity.
    box.velocity = np.array((0.0, 0.0, 0.0))

    # # Optional: Filter by max_dist
    # if max_dist is not None:
    #     dist = np.sqrt(np.sum(box.center[:2] ** 2))
    #     if dist > max_dist:
    #         continuen

    return np.concatenate((box.center, box.wlh[[1,0,2]], np.array([quaternion_yaw(box.orientation)]), box.velocity[:2]), axis=0).reshape(-1,9)    # Note: nuscenes_boxes_lidar_pred is lwh but nuscenes_lidar is wlh



# def eval(cfg, model, validation_loader, counter, histo_counter, epoch, writer, rank, lbl):
#     #cfg
#     home_path = '/home/hpaat/my_exp/MTrans-evidential'
#     data_path = '/data/kitti_detect'
#     val_link_path = '/home/hpaat/pcdet/data/nuscenes/kitti_ver/val/val_link.txt'
#     nusc = cfg.dataset == 'NUSCENES'
    
#     if nusc:
#         nuscenes = NuScenes(dataroot='/home/hpaat/pcdet/data/nuscenes/v1.0-trainval', version='v1.0-trainval')
#         id_to_token = {}
#         with open(val_link_path) as f:
#             val_link = f.readlines()
#         for line in val_link:
#             token, id = line.split(',')
#             id_to_token[id.strip()] = token 
#     else:
#         id_to_token = None

#     model.eval()
#     process_bar = tqdm(validation_loader, desc='Evaluate model')
#     counter.reset()
#     histo_counter.reset()
#     all_nuscenes_boxes = {}
#     uncertaintys = torch.Tensor([])
#     frame_list = []

#     with torch.no_grad():
#         det_annos = []

#         # Modified/Changed/Added by HP
#         # Get access to train/val IDs
#         if not nusc:
#             with open(home_path + data_path + '/ImageSets/' + lbl + '.txt') as f: # change this to train.txt if validation loader uses split="train"
#                 id_list = f.readlines()
#             id_list = [id.strip() for id in id_list]
#             id_to_lidar_path = None
#         elif nusc and lbl=='val': 
#             id_list = [f"{id:06d}" for id in range(6019)]
            
#             id_to_lidar_path = {}
#             for id in id_list:
#                 sample_token = id_to_token[id]
#                 sample = nuscenes.get('sample', sample_token)
#                 lidar_token = sample['data']['LIDAR_TOP']
#                 sd_record_lid = nuscenes.get('sample_data', lidar_token)
#                 filename_lid_full = sd_record_lid['filename']
#                 id_to_lidar_path[id] = filename_lid_full.split('/')[-1] 
        
#         ######################################################################################################################################################
        
#         for data in process_bar:
#             data = EasyDict(data)

#             data = move_to_cuda(data, 'cuda', rank)
#             pred_dict = model(data)
#             if cfg.dist or cfg.is_dp: 
#                 loss_dict, loss, iou3d_histo, loss_box = model.module.get_loss(pred_dict, data, rank)   # model.get_loss(pred_dict, data)
#             else:
#                 loss_dict, loss, iou3d_histo, loss_box = model.get_loss(pred_dict, data, rank)          # model.get_loss(pred_dict, data)
#             var = pred_dict['conf'].view(-1)
#             uncertaintys = torch.cat([uncertaintys, var.detach().cpu()], dim=0)
#             frame_list.extend(data['frames'])

#             if cfg.dataset=='KITTI':

#                 # Get the label as what is printed on the txt file
#                 if lbl=="train":
#                     label, frames = format_kitti_labels(pred_dict, data, with_score=(validation_loader.dataset.cfg.split=='train'), nusc=True)
#                 elif lbl=="val":
#                     label, frames = format_kitti_labels(pred_dict, data, with_score=(validation_loader.dataset.cfg.split=='val'), nusc=True)
                                
#                 boxes_lidar_nusc = get_boxes_lidar_nuscenes_format(label, frames, lbl) if nusc else None

#                 # Define annos variable
#                 annos = get_annos_dict(label, frames, pred_dict, nusc, id_to_token, id_to_lidar_path, boxes_lidar_nusc)
                
#                 if nusc:
#                     if len(det_annos) == 0 or data.frames[0] != det_annos[-1]["frame_id_kitti_ver"]:
#                         det_annos.append(annos)                     
#                     else: 
#                         # No need to include additional element to det_annos (just append to the last element)
#                         det_annos[-1]['name'] = np.concatenate((det_annos[-1]["name"], annos['name']), axis=0)
#                         det_annos[-1]['score'] = np.concatenate((det_annos[-1]["score"], annos['score']), axis=0)
#                         det_annos[-1]['boxes_lidar'] = np.concatenate((det_annos[-1]["boxes_lidar"], annos['boxes_lidar']), axis=0) 
#                         det_annos[-1]['pred_labels'] = np.concatenate((det_annos[-1]["pred_labels"], annos['pred_labels']), axis=0)
                        
#                     while det_annos[-1]["frame_id_kitti_ver"] != id_list[len(det_annos)-1]:
#                         if cfg.gen_label:
#                             if not path.exists(f'{cfg.label_dir}'):
#                                 makedirs(f'{cfg.label_dir}')
#                             with open(path.join(f'{cfg.label_dir}', f'{id_list[len(det_annos)-1]}.txt'), 'a') as f:
#                                 #l = label[i]
#                                 # score = float(l.split(' ')[-1])       # [optional]: discard 3D predictions with low confidence
#                                 # if score<0.05:
#                                 #     continue
#                                 f.write('')  
                            
#                         num_samples = 0
#                         no_content_id = id_list[len(det_annos)-1]
#                         det_annos.append({
#                             'name': np.zeros(num_samples), 'score': np.zeros(num_samples), 
#                             'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples),
#                             'frame_id_kitti_ver': no_content_id, 'frame_id': id_to_lidar_path[no_content_id],
#                             'metadata': {'token': id_to_token[no_content_id]}
#                         })
#                         det_annos[-1], det_annos[-2] = det_annos[-2], det_annos[-1]

#                 else:
#                     if len(det_annos) == 0 or data.frames[0] != det_annos[-1]["frame_id"]:
#                         det_annos.append(annos)                     
#                     else: 
#                         # No need to include additional element to det_annos (just append to the last element)
#                         det_annos[-1]['name'] = np.concatenate((det_annos[-1]["name"], annos['name']), axis=0)
#                         det_annos[-1]['truncated'] = np.concatenate((det_annos[-1]["truncated"], annos['truncated']), axis=0)  
#                         det_annos[-1]['occluded'] =  np.concatenate((det_annos[-1]["occluded"], annos['occluded']), axis=0) 
#                         det_annos[-1]['alpha'] = np.concatenate((det_annos[-1]["alpha"], annos['alpha']), axis=0)  
#                         det_annos[-1]['bbox'] = np.concatenate((det_annos[-1]["bbox"], annos['bbox']), axis=0) 
#                         det_annos[-1]['dimensions'] = np.concatenate((det_annos[-1]["dimensions"], annos['dimensions']), axis=0) 
#                         det_annos[-1]['location'] = np.concatenate((det_annos[-1]["location"], annos['location']), axis=0) 
#                         det_annos[-1]['rotation_y'] = np.concatenate((det_annos[-1]["rotation_y"], annos['rotation_y']), axis=0) 
#                         det_annos[-1]['score'] = np.concatenate((det_annos[-1]["score"], annos['score']), axis=0) 
#                         det_annos[-1]['boxes_lidar'] = np.concatenate((det_annos[-1]["boxes_lidar"], annos['boxes_lidar']), axis=0)
                
#                     while det_annos[-1]["frame_id"] != id_list[len(det_annos)-1]:
#                         if cfg.gen_label:
#                             if not path.exists(f'{cfg.label_dir}'):
#                                 makedirs(f'{cfg.label_dir}')
#                             with open(path.join(f'{cfg.label_dir}', f'{id_list[len(det_annos)-1]}.txt'), 'a') as f:
#                                 #l = label[i]
#                                 # score = float(l.split(' ')[-1])       # [optional]: discard 3D predictions with low confidence
#                                 # if score<0.05:
#                                 #     continue
#                                 f.write('')  
                            
#                         num_samples = 0
#                         det_annos.append({
#                             'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
#                             'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
#                             'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
#                             'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
#                             'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
#                             'frame_id': id_list[len(det_annos)-1]
#                         })
#                         det_annos[-1], det_annos[-2] = det_annos[-2], det_annos[-1]
                    
#                 ################################################################################################################################################ 
#                 if cfg.gen_label:
#                     if not path.exists(f'{cfg.label_dir}'):
#                         makedirs(f'{cfg.label_dir}')
#                     for i, fr in enumerate(frames):
#                         with open(path.join(f'{cfg.label_dir}', f'{fr}.txt'), 'a') as f:
#                             l = label[i]
#                             f.write(l+'\n')  
    
#             # Statistics
#             counter.update(loss_dict)
#             histo_counter.update(iou3d_histo)

#             pbar_text = get_pbar_text(counter, f'Eval')
#             process_bar.set_description(pbar_text)

#         # If the final frames of nusc have no content
#         if nusc:
#             if len(det_annos) != len(id_list):
#                 last_id_with_content = int(det_annos[-1]['frame_id_kitti_ver']) # 6015
#                 while last_id_with_content < len(id_list)-1:
#                     last_id_with_content = last_id_with_content+1
#                     no_content_id = f"{last_id_with_content:06d}"
#                     det_annos.append({
#                             'name': np.zeros(num_samples), 'score': np.zeros(num_samples), 
#                             'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples),
#                             'frame_id_kitti_ver': no_content_id, 'frame_id': id_to_lidar_path[no_content_id],
#                             'metadata': {'token': id_to_token[no_content_id]}
#                         })
        
#         # Save det_annos for external evaluation in pcdet
#         save_det_annos = True
#         if save_det_annos:
#             with open(home_path + '/output/' + cfg.experiment_name + '/det_annos_' + str(epoch) + '_' + lbl + '.pkl', 'wb') as f: 
#                 pickle.dump(det_annos, f)

#         stats = counter.average(None, group_by_description=True)
#         for group in stats.keys():
#             writer.add_scalars(f'Eval/{group}', stats[group], epoch)
#         writer.add_histogram('Eval/iou_distribution', histo_counter.get_values(), epoch)
        
#         # metric for saving best checkpoint
#         score = (counter.average(['iou3d'])['iou3d'])

#     return score

def main(rank, num_gpus, cfg, cfg_path):
    # import pdb; pdb.set_trace() 
    torch.cuda.set_device(rank)
    if cfg.dist:
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=num_gpus, init_method='env://')

    freeze_random_seed(cfg.random_seed)

    # Tensorboard, Yaml Config, Score Counter
    output_path = path.join(cfg.TRAIN_CONFIG.output_root, cfg.experiment_name)
    writer = SummaryWriter(log_dir=path.join(output_path, cfg.experiment_name+'_tb'))
    writer.add_text('experiment_name', cfg.experiment_name, 0)
    writer.add_text('start_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0)
    counter = ScoreCounter()
    histo_counter = HistoCounter()
    
    # Save config file
    shutil.copyfile(cfg_path, os.path.join(output_path, 'MTrans_kitti_' + str(datetime.now()) + '.yaml'))

    # Config files
    train_cfg, dataset_cfg, loader_cfg = cfg.TRAIN_CONFIG, cfg.DATASET_CONFIG, cfg.DATALOADER_CONFIG
    data_root = cfg.data_root

    # Build dataset and dataloader
    if cfg.dataset == 'KITTI':
        nusc = False
        dataset = KittiDetectionDataset
        loader_builder = build_kitti_loader
    elif cfg.dataset == 'NUSCENES':
        nusc = True
        dataset = KittiDetectionDataset
        loader_builder = build_kitti_loader
    else:
        raise RuntimeError
    
    # Training Set
    training_set = dataset(data_root, dataset_cfg.TRAIN_SET, labeled=True, nusc=nusc)
    training_loader = loader_builder(training_set, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus)
    train_length = len(training_loader)

    # Training Set for Label Generation (includes complete training set)
    training_set_for_gen_label = dataset(data_root, dataset_cfg.TRAIN_SET, gen_pseudolabel=cfg.gen_label, nusc=nusc)
    training_loader_for_gen_label = loader_builder(training_set_for_gen_label, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus)

    # Unlabeled Training Set
    # if loader_cfg.TRAIN_LOADER.unsupervise_batch_size > 0:
    #     # build another dataset that has no 3D label
    #     temp_cfg = deepcopy(dataset_cfg.TRAIN_SET)
    #     temp_cfg.use_3d_label = False
        
    #     unlabeled_training_set = dataset(data_root, temp_cfg, nusc=nusc)
    #     # Training loader for unlabeled dataset
    #     temp_cfg = deepcopy(loader_cfg.TRAIN_LOADER)
    #     temp_cfg.batch_size = temp_cfg.unsupervise_batch_size
    #     unlabeled_training_loader = loader_builder(unlabeled_training_set, cfg, temp_cfg, rank, num_gpus)
    # else:
    #     temp_cfg = None
    #     unlabeled_training_set = None
    #     unlabeled_training_loader = None

    temp_cfg = None
    unlabeled_training_set = dataset(data_root, dataset_cfg.TRAIN_SET, labeled=False, nusc=nusc)
    
    # Validation Set
    validation_set = dataset(data_root, dataset_cfg.VAL_SET, nusc=nusc)
    validation_loader = loader_builder(validation_set, cfg, loader_cfg.VAL_LOADER, rank, num_gpus)
    
    # Build the Model
    model = MTrans(cfg.MODEL_CONFIG)
    model.cuda(rank)
    if cfg.dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[rank])
    elif cfg.is_dp:
        assert cfg.dist is False
        model = torch.nn.DataParallel(model)

    print("[Model Params]: ", sum([p.numel() for n, p in model.named_parameters()]))
    writer.add_text('model_params', str(sum([p.numel() for n, p in model.named_parameters()])))

    # Build optimizer and lr_scheduler
    if model.decay is not None:
        optim = getattr(torch.optim, train_cfg.optimizer)(lr=train_cfg.lr, params=model.parameters(), weight_decay=model.decay)
    else:
        optim = getattr(torch.optim, train_cfg.optimizer)(lr=train_cfg.lr, params=model.parameters())
    scheduler = WarmupCosineAnnealing(optim, train_cfg.lr, train_cfg.warmup_rate, train_cfg.epochs*train_length, eta_min=0)
    scheduler.step()
    
    # Load checkpoint (if any)
    start_epoch=0

    if cfg.init_checkpoint is not None:
        print(f"Loading checkpoint at: {cfg.init_checkpoint}")
        start_epoch = load_checkpoint(f'{cfg.init_checkpoint}', model, optim, scheduler) + 1
    elif path.exists(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/best_model.pt'):       # But if best_model.pt exists
        print("Loading best checkpoints...")
        start_epoch = load_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/best_model.pt', model, optim, scheduler) + 1
    run = runner(cfg)

    # NOTE eval function has setting to generate pseudo labels
    if cfg.gen_label:                           # start_epoch > 0:
        # pass
        training_score = run.eval(cfg, model, training_loader_for_gen_label, counter, histo_counter, start_epoch-1, writer, rank, lbl="train", gen_label_prints=True)     # Changes made by Helbert      # Just for saving training pseudolabels
        import pdb; pdb.set_trace() 
        best_score = run.eval(cfg, model, validation_loader, counter, histo_counter, start_epoch-1, writer, rank, lbl="val", gen_label_prints=True)                       # Save det_annos & pseudolabels  
        print("For label generation, run only until this point")
        import pdb; pdb.set_trace() 

    # Initially train with the training dataset. Set unlabeled_training_set to None.
    run.run(loader_builder, training_set, None, validation_set, start_epoch, cfg, train_cfg, loader_cfg, temp_cfg, model, optim, scheduler, counter, histo_counter, writer, rank, num_gpus, init_run=True)
    
    import pdb; pdb.set_trace() 
    episodes = 30
    uncertainty_type = 'conf'
    for episode_cnt in range(episodes):
        # Use the trained model to predict for the entire training dataset      # TODO loss_weight of 1.0 for those labeled dataset
        # NOTE gen_pseudolabel set to True to get ALL training set
        training_set_for_pred = dataset(data_root, dataset_cfg.TRAIN_SET, gen_pseudolabel=True, nusc=nusc)
        unlabeled_training_set, uncertaintys = run.predict(cfg, loader_builder, model, training_set_for_pred, loader_cfg, 
                                                           counter, histo_counter, rank, num_gpus, writer, optim, scheduler,
                                                           uncertainty_type = uncertainty_type) 

        # Use the trained model to predict for the unlabeled dataset
        # unlabeled_training_set, uncertaintys = run.predict(cfg, loader_builder, model, unlabeled_training_set, loader_cfg, 
        #                                                    counter, histo_counter, rank, num_gpus, writer, optim, scheduler,
        #                                                    uncertainty_type = uncertainty_type)                                 # NOTE Output the uncertainties in the order of the objects, NOT shuffled  
        #        
        assert len(uncertaintys) == len(unlabeled_training_set), "len(uncertaintys) must equal len(unlabeled_training_set)"
        
        # Define the loss weights using the output uncertaintys of the model and normalize across the entire unlabeled training set
        loss_weight = uncertaintys.cpu().clone().detach().reshape(-1)
        
        import pdb; pdb.set_trace() 
        min_val = 1e-5
        if uncertainty_type != 'conf' or uncertainty_type == 'gt_iou': loss_weight = 1 / (loss_weight + min_val)                # If conf, loss_weight should be uncertaintys
        loss_weight = loss_weight/sum(loss_weight)

        ignore_uncertainty = False
        if ignore_uncertainty:
            loss_weight = torch.tensor(len(loss_weight) * [1.0])
            loss_weight = loss_weight/sum(loss_weight)
        
        # Update unlabeled_training_set object weights. 
        unlabeled_training_set.update_weights(loss_weight)

        # loss = torch.tensor(n, 1) # n number of object
        # weighted_loss = weight * loss
        # weighted_loss = torch.sum(loss) / torch.sum(weighted_loss) * weighted_loss

        # Train using the combined dataset --> Just make the unlabeled dataset not none
        run.run(loader_builder, training_set, unlabeled_training_set, validation_set, start_epoch, cfg, train_cfg, loader_cfg, temp_cfg, model, optim, scheduler, counter, histo_counter, writer, rank, num_gpus, episode_num=episode_cnt) 

    writer.flush()
    writer.close()
    if cfg.dist:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    
    num_gpus = torch.cuda.device_count()
    
    parser = argparse.ArgumentParser(description='training arguments')
    parser.add_argument('--cfg_file', type=str, help='the path to configuration file')
    args = parser.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))

    if cfg.dist:
        # Added multi-gpu training with MultipleProcesses by Helbert PAAT
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    print("===== START TRAINING =====")
    print(cfg.experiment_name)
    print("==========================")

    if cfg.dist:
        torch.multiprocessing.spawn(main, args=(num_gpus, cfg, args.cfg_file), nprocs=num_gpus, join=True) # modified/changed by Helbert PAAT to include multi-gpu feature
    elif cfg.is_dp:
        assert cfg.dist is False
        main(0, num_gpus, cfg, args.cfg_file)
    else:
        main(0, num_gpus, cfg, args.cfg_file)