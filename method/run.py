import torch
from tqdm import tqdm
from datasets.kitti_loader import build_kitti_loader, move_to_cuda, merge_two_batch, make_tensor_keys
from os import path, makedirs
import pickle
import numpy as np
from easydict import EasyDict
import math
import torch.distributed as dist
import os
from loss import cal_iou_3d

from nuscenes.nuscenes import NuScenes
import scipy
import random

from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.data_classes import Box

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_checkpoint(save_path, epoch, model, optim=None, scheduler=None):
    if is_main_process():

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

def format_kitti_labels(pred_dict, data_dict, with_score=True):
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

            if 'scores' in data_dict.keys() and sum(data_dict['scores']) != 0:
                # for test result, MAPGen confidence * 2D Box score
                score = score * data_dict['scores'][i] / max(pred_dict['conf']).item()

            if with_score:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f} {score:.4f}')
            else:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f}')
                
        return labels, data_dict.frames

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

def get_annos_dict(labels, frame_id, pred_dict, nusc, id_to_token, id_to_lidar_path, boxes_lidar_nusc):
    # TODO Should cater any batch size
    # This only caters to batch_size of 1
    try: name, truncated, occluded, alpha, bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4, dimensions_1, dimensions_2, \
    dimensions_3, location_1, location_2, location_3, rotation_y, score = labels[0].split()
    except: name, truncated, occluded, alpha, bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4, dimensions_1, dimensions_2, \
    dimensions_3, location_1, location_2, location_3, rotation_y = labels[0].split()
        
    name = np.array([name])
    truncated = np.array([float(truncated)], dtype=np.float32)
    occluded = np.array([occluded], dtype=np.float32)
    alpha = np.array([alpha], dtype=np.float32)
    bbox = np.array([[bbox2d_1, bbox2d_2, bbox2d_3, bbox2d_4]], dtype=np.float32)
    dimensions = np.array([[dimensions_3, dimensions_1, dimensions_2]], dtype=np.float32)
    location = np.array([[location_1, location_2, location_3]], dtype=np.float32)
    rotation_y = np.array([rotation_y], dtype=np.float32)
    try: score = np.array([score], dtype=np.float32)
    except: score = np.array([0], dtype=np.float32)
    
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

def read_label(label, calib, classes=['Car']):
    labels = [label.strip().split(' ')]     # A list of list

    for label in labels:
        if label[0] == 'DontCare':
            continue
        if label[0] not in classes:
            continue
        cls = label[0]
        truncated = float(label[1])
        if truncated > 0.95:            # remove too much truncated
            continue

        occluded = int(label[2])
        box_2d = np.array(label[4:8], dtype=np.float32)
        dim = np.array(label[8:11], dtype=np.float32)
        loc = np.array(label[11:14], dtype=np.float32)
        yaw = float(label[14])

        # Change label coordinate system: camera sys -> lidar sys
        location = calib.rect_to_velo(loc[np.newaxis, ...])
        x, y, z = location[0]
        h, w, l = dim
        z += h/2
        yaw = -yaw - np.pi/2

        object_labels = {
            'class': cls,
            'truncated': truncated,
            'occluded': occluded,
            'box_2d': box_2d,
            'dimensions': np.array([l, w, h]),
            'location': np.array([x, y, z]),
            'yaw': yaw,
        }

        if len(label)==16:
            score = float(label[15])
            object_labels['score'] = score

    return object_labels

def get_pred_evidential_aleatoric(out):
    v, alpha, beta = out
    var = beta / (alpha - 1)
    return torch.mean(var, dim=1)

def get_pred_evidential_epistemic(out):
    v, alpha, beta = out
    var = beta / (v * (alpha - 1))
    return torch.mean(var, dim=1)

class runner(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.prog_metric_dir = cfg.prog_metric_dir
        self.prog_metric_dir_corr = cfg.prog_metric_dir + '_corr.txt'
        self.evi_uncertainty = cfg.MODEL_CONFIG.evi_uncertainty.setting
        self.ensemble = cfg.MODEL_CONFIG.ensemble
        self.mcdo = cfg.MODEL_CONFIG.mcdo
        self.debug = cfg.DEBUG
        self.gen_pseudo_from_external = cfg.gen_pseudo_from_external
        try: self.gen_pseudo_from_external_path = cfg.gen_pseudo_from_external_path
        except: self.gen_pseudo_from_external_path = None

        self.home_path = cfg.home_path
        self.data_path = cfg.data_root[1:]        #'/data/kitti_detect'
        self.val_link_path = cfg.val_link_path    

        self.ckpt_save_interval = cfg.TRAIN_CONFIG.ckpt_save_interval
        self.val_see_perf_limit = cfg.TRAIN_CONFIG.val_see_perf_limit
        self.save_det_annos = cfg.save_det_annos

        self.conf_save_interval = cfg.MODEL_CONFIG.evi_uncertainty.conf_save_interval
        self.conf_dir = '/'.join(cfg.prog_metric_dir.split('/')[:2]) + '/' + cfg.experiment_name + '/conf/'
        if not os.path.isdir(self.conf_dir):
            os.makedirs(self.conf_dir)

    def get_pbar_text(self, counter, prefix, prog_save_folder, gen_label_prints=False):
        #stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf', 'evidential_loss'])
        #pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, evi_loss:{stats['evidential_loss']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}"
         
        if self.evi_uncertainty:
            stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf' \
                                    , 'evidential_loss', 'evi_iou_corr', 'evi_iou_corr_epis',  #, 'v', 'alpha', 'beta' 
                                #'rot_loss'
                                    ])
            
            pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}, \
                        evi_loss: {stats['evidential_loss']:.2f}, evi_iou_corr: {stats['evi_iou_corr']*100:.2f}, evi_iou_corr_epis: {stats['evi_iou_corr_epis']*100:.2f}" #, rot_loss: {stats['rot_loss']*100:.2f}"        #, v: {stats['v']*100:.2f}, alpha: {stats['alpha']*100:.2f}, beta: {stats['beta']*100:.2f}"
        
        elif self.ensemble:
            stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf' \
                                    , 'ensemble_var_checker'])

            pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}, Ens_var: {stats['ensemble_var_checker']*100:.2f}" #, rot_loss: {stats['rot_loss']*100:.2f}"        #, v: {stats['v']*100:.2f}, alpha: {stats['alpha']*100:.2f}, beta: {stats['beta']*100:.2f}"

        else:
            stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf' \
                                    ])

            pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}"

        # Printing a few results    # Changes - Helbert
        if random.random() < 0.20:
            if gen_label_prints:
                prog_save_folder += '_genlabel.txt'
                # if path.isfile(prog_save_folder):
                #     os.remove(prog_save_folder)
            with open(prog_save_folder, "a") as file:
                file.write(pbar_text + "\n")
            
        return pbar_text

    def run(self, loader_builder, training_set, unlabeled_training_set, validation_set, start_epoch, cfg, train_cfg, loader_cfg, temp_cfg, model, optim, scheduler, counter, histo_counter, writer, rank, num_gpus, episode_num = None, init_run=False):
        
        # Define loaders
        training_loader = loader_builder(training_set, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus)
        validation_loader = loader_builder(validation_set, cfg, loader_cfg.VAL_LOADER, rank, num_gpus)

        if unlabeled_training_set is not None:
            unlabeled_training_loader = loader_builder(unlabeled_training_set, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus)                # loaders must all be shuffled
        else:
            unlabeled_training_loader = None

        if init_run:
            num_epochs = train_cfg.init_epochs
        else:
            num_epochs = train_cfg.epochs_per_episode
            start_epoch = 0

        best_score = -9999
        for epoch in range(start_epoch, num_epochs):
            if cfg.dist:
                training_loader.sampler.set_epoch(epoch)
                unlabeled_training_loader.sampler.set_epoch(epoch)

            # TRAINING    
            self.train_one_epoch(cfg, model, training_loader, unlabeled_training_loader, optim, scheduler, counter, histo_counter, epoch, writer, rank)
            # if start_epoch < train_cfg.epochs and ((epoch+1) % train_cfg.epoches_per_eval) == 0 and epoch >= 10:
            # # save last checkpoint
            #     save_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/epoch_{epoch}.pt', epoch, model, optim, scheduler)
            
            # EVALUATION
            actual_epoch = (episode_num * train_cfg.epochs_per_episode) + epoch + train_cfg.init_epochs if episode_num is not None else epoch
           
            if ((actual_epoch + 1) % train_cfg.epoches_per_eval) == 0 or \
                ((actual_epoch + 1) % self.ckpt_save_interval == 0 and actual_epoch + 1 >= train_cfg.eval_begin):
                if (actual_epoch + 1) % self.ckpt_save_interval == 0 and actual_epoch + 1 >= train_cfg.eval_begin:
                    see_perf = False
                else:
                    see_perf = True
                # import pdb; pdb.set_trace()
                score = self.eval(cfg, model, validation_loader, counter, histo_counter, actual_epoch, writer, rank, lbl="val", see_perf=see_perf)         # may save det_annos & pseudolabels     
                if score > best_score and (actual_epoch + 1) % self.ckpt_save_interval == 0 and actual_epoch + 1 >= train_cfg.eval_begin:
                    best_score = score
                    save_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/best_model_{str(actual_epoch)}.pt', actual_epoch, model, optim, scheduler)

        # if start_epoch < train_cfg.epochs and rank==0:
        #     # Save last checkpoint
        #     save_checkpoint(f'{cfg.TRAIN_CONFIG.output_root}/{cfg.experiment_name}/ckpt/epoch_{epoch}.pt', epoch, model, optim, scheduler)

    def train_one_epoch(self, cfg,
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
        
        # Initialize empty numpy variables 
        iou3d, evi_unc, evi_unc_epis, v, alpha, beta, pred_boxes_all, gt_boxes_all = np.array([]), np.array([]), np.array([]),  np.array([]),  np.array([]),  np.array([]), np.array([]),  np.array([])

        for data in process_bar:
            optim.zero_grad()
            if unlabeled_training_loader is not None:
                try:
                    unlabeled_data = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_training_loader)
                    unlabeled_data = next(unlabeled_iter)
                data = merge_two_batch(data, unlabeled_data)
            #data = make_tensor_keys(data)
            data = move_to_cuda(data, 'cuda', rank)
            
            pred_dict = model(data)

            weights = [float(i) for i in data['weights']]
            if cfg.dist or cfg.is_dp:
                loss_dict, loss, iou3d_histo, loss_box= model.module.get_loss(pred_dict, data, rank, weights)
            else:
                if self.evi_uncertainty:
                    loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, evi_unc_addl, evi_unc_addl_epistemic, v_addl, alpha_addl, beta_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank, weights)
                elif self.ensemble:
                        loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes, var = model.get_loss(pred_dict, data, rank) 
                elif self.mcdo:
                    loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank)   
                else:        
                    loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank)  

            histo_counter.update(iou3d_histo)

            if self.evi_uncertainty:
                # Append values
                iou3d = np.append(iou3d, iou3d_addl)
                evi_unc = np.append(evi_unc, evi_unc_addl)
                evi_unc_epis = np.append(evi_unc_epis, evi_unc_addl_epistemic)
                try: v = np.concatenate((v, v_addl), axis=0)
                except: v = v_addl
                try: alpha = np.concatenate((alpha, alpha_addl), axis=0)
                except: alpha = alpha_addl
                try: beta = np.concatenate((beta, beta_addl), axis=0)
                except: beta = beta_addl
                try: pred_boxes_all = np.concatenate((pred_boxes_all, pred_boxes), axis=0)
                except: pred_boxes_all = pred_boxes
                try: gt_boxes_all = np.concatenate((gt_boxes_all, gt_boxes), axis=0)
                except: gt_boxes_all = gt_boxes
            
            # statistics
            counter.update(loss_dict)
            loss.backward()
            
            optim.step()
            scheduler.step()
            counter.update({'lr':(optim.param_groups[0]['lr'], 1, 'learning_rate')})

            pbar_text = self.get_pbar_text(counter, f'T-{epoch}', self.prog_metric_dir)        
            process_bar.set_description(pbar_text)
        
        # Save values on text file for visualization. Changes - Helbert
        if self.evi_uncertainty:
            alea = beta / (alpha - 1)
            epis = beta / (v * (alpha - 1))
            alea_mean =  np.mean(alea, axis=0)
            v_mean = np.mean(v, axis=0)
            alpha_mean = np.mean(alpha, axis=0)
            epis_mean = np.mean(epis, axis=0)
            beta_mean = np.mean(beta, axis=0)

            confidence = np.sqrt(1. / ((alpha-1) * v))      # sqrt of the inverse evidence
            conf_mean = confidence.mean(axis=0)

            gt_std = gt_boxes_all.std(axis=0)
            pred_std = pred_boxes_all.std(axis=0)
            gt_mean = gt_boxes_all.mean(axis=0)
            pred_mean = pred_boxes_all.mean(axis=0)
            res_mean = (gt_boxes_all-pred_boxes_all).mean(axis=0)
            res_std = (gt_boxes_all-pred_boxes_all).std(axis=0)

            dim = ['x', 'y', 'z', 'l', 'w', 'h', 'rot']
            with open(self.prog_metric_dir_corr, "a") as file:
                file.write(f"T-{epoch} {str(scipy.stats.pearsonr(iou3d, evi_unc)[0])}, evi_unc_alea_spear:{str(scipy.stats.spearmanr(iou3d, evi_unc)[0])}, ")
                for i in range(7):
                    file.write(f"alea_corr_iou_{dim[i]}:{str(scipy.stats.pearsonr(iou3d, alea[:,i])[0])}, ")

                # confidence
                for i in range(7):
                    file.write(f"conf_{dim[i]}:{str(conf_mean[i])}, ")
                for i in range(7):
                    file.write(f"conf_iou_corr_{dim[i]}:{str(scipy.stats.pearsonr(iou3d, confidence[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"res_unc_corr_{dim[i]}:{str(scipy.stats.pearsonr(alea[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"res_epis_corr_{dim[i]}:{str(scipy.stats.pearsonr(epis[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"res_conf_corr_{dim[i]}:{str(scipy.stats.pearsonr(confidence[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"v_{dim[i]}:{str(v_mean[i])}, ")
                for i in range(7):
                    file.write(f"alpha_{dim[i]}:{str(alpha_mean[i])}, ")
                for i in range(7):
                    file.write(f"beta_{dim[i]}:{str(beta_mean[i])}, ")
                for i in range(7):
                    file.write(f"alea_{dim[i]}:{str(alea_mean[i])}, ")
                for i in range(7):
                    file.write(f"epis_{dim[i]}:{str(epis_mean[i])}, ")
                for i in range(7):
                    file.write(f"gt_std_{dim[i]}:{str(gt_std[i])}, ")
                for i in range(7):
                    file.write(f"pred_std_{dim[i]}:{str(pred_std[i])}, ")
                for i in range(7):
                    file.write(f"gt_mean_{dim[i]}:{str(gt_mean[i])}, ")
                for i in range(7):
                    file.write(f"pred_mean_{dim[i]}:{str(pred_mean[i])}, ")
                for i in range(7):
                    file.write(f"res_mean_{dim[i]}:{str(res_mean[i])}, ")
                for i in range(7):
                    file.write(f"res_std_{dim[i]}:{str(res_std[i])}, ")
                file.write("\n")

            with open(self.prog_metric_dir_corr + 'epis.txt', "a") as file:
                file.write(f"T-{epoch} evi_unc_epis:{str(scipy.stats.pearsonr(iou3d, evi_unc_epis)[0])}, evi_unc_epis_spear:{str(scipy.stats.spearmanr(iou3d, evi_unc_epis)[0])}, ")
                
                for i in range(7):
                    file.write(f"epis_corr_iou_{dim[i]}:{str(scipy.stats.pearsonr(iou3d, epis[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"epis_corr_iou_spear_{dim[i]}:{str(scipy.stats.spearmanr(iou3d, epis[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"alea_corr_iou_spear_{dim[i]}:{str(scipy.stats.spearmanr(iou3d, alea[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"conf_iou_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(iou3d, confidence[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"res_unc_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(alea[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"res_epis_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(epis[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                for i in range(7):
                    file.write(f"res_conf_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(confidence[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                file.write("\n")

            # Show in terminal
            print("Pearson evi all: " + str(scipy.stats.pearsonr(iou3d, evi_unc)[0]))
            print("Pearson epis all: " + str(scipy.stats.pearsonr(iou3d, evi_unc_epis)[0]))
            print("Spearman evi all: " + str(scipy.stats.spearmanr(iou3d, evi_unc)[0]))
            print("Spearman epis all: " + str(scipy.stats.spearmanr(iou3d, evi_unc_epis)[0]))
            
            # Save confidence, pred, target, uncertainties
            if (epoch+1) % self.conf_save_interval == 0:                
                with open(self.conf_dir + 'conf_' + str(epoch) + '_train.pkl', 'wb') as file:
                    pickle.dump({'pred': pred_boxes_all, 'target': gt_boxes_all, 'conf': confidence, 'error': pred_boxes_all-gt_boxes_all, 'alea':alea, 'beta':beta, 'alpha':alpha, 'v':v, 'iou3d': iou3d}, file)

        stats = counter.average(None, group_by_description=True)
        for group in stats.keys():
            writer.add_scalars(f'Train/{group}', stats[group], epoch)
        writer.add_histogram('Train/iou_distribution', histo_counter.get_values(), epoch)
        
    def eval(self, cfg, model, validation_loader, counter, histo_counter, epoch, writer, rank, lbl, gen_label_prints=False, see_perf=False):

        nusc = (cfg.dataset == 'NUSCENES')
        
        # Set to False by default, only used to generate pseudo labels from Ensemble and MC Dropout when the predicted dict loc, dim, and yaw are from an externally saved file
        if self.gen_pseudo_from_external and lbl=='train':
            print("GENERATING PSEUDO FROM EXTERNAL PKL")
            with open(self.gen_pseudo_from_external_path, "rb") as f:
                ensemble_pseudo = pickle.load(f)
                ensemble_pseudo = torch.from_numpy(ensemble_pseudo).cuda()
                print("NOTE: the pred_dict is from external. \n")
                import pdb; pdb.set_trace()
        
        if nusc:
            nuscenes = NuScenes(dataroot='/home/hpaat/pcdet/data/nuscenes/v1.0-trainval', version='v1.0-trainval')
            id_to_token = {}
            if lbl == 'val':
                with open(self.val_link_path) as f:
                    val_link = f.readlines()
                for line in val_link:
                    token, id = line.split(',')
                    id_to_token[id.strip()] = token 
            elif lbl == 'train':
                with open("/home/hpaat/pcdet/data/nuscenes/kitti_ver/train/train_link.txt") as f:       # TODO Add to config
                    train_link = f.readlines()
                for line in train_link:
                    token, id = line.split(',')
                    id_to_token[id.strip()] = token 
        else:
            id_to_token = None

        model.eval()
        process_bar = tqdm(validation_loader, desc='Evaluate model')
        counter.reset()
        histo_counter.reset()
        all_nuscenes_boxes = {}
        
        with torch.no_grad():
            det_annos = []

            # Get access to train/val IDs - Modified/Changed/Added by HP      
            if not nusc:
                with open(self.home_path + self.data_path + '/ImageSets/' + lbl + '.txt') as f:
                    id_list = f.readlines()
                id_list = [id.strip() for id in id_list]
                id_to_lidar_path = None
            elif nusc:
                if lbl=='val': 
                    size = 6019
                elif lbl=='train': 
                    size = 28130
                id_list = [f"{id:06d}" for id in range(size)]
                id_to_lidar_path = {}
                for id in id_list:
                    sample_token = id_to_token[id]
                    sample = nuscenes.get('sample', sample_token)
                    lidar_token = sample['data']['LIDAR_TOP']
                    sd_record_lid = nuscenes.get('sample_data', lidar_token)
                    filename_lid_full = sd_record_lid['filename']
                    id_to_lidar_path[id] = filename_lid_full.split('/')[-1]
            
            ######################################################################################################################################################
            # Initialize empty numpy variables
            iou3d, evi_unc, evi_unc_epis, v, alpha, beta, pred_boxes_all, gt_boxes_all, var_all = np.array([]), np.array([]), np.array([]),  np.array([]),  np.array([]),  np.array([]), np.array([]),  np.array([]), np.array([])
            
            for idx, data in enumerate(process_bar):
                # See performance only on the first few objects and not on the entire process bar objects
                if see_perf and (idx+1)*len(data['frames']) >= self.val_see_perf_limit:
                    break
                
                data = EasyDict(data)
                data = move_to_cuda(data, 'cuda', rank)

                # Get prediction from trained model
                pred_dict = model(data)

                if self.gen_pseudo_from_external and lbl=='train':
                    # Disregards the actual pred_dict and update pred_dict from an external file
                    box_for_update = ensemble_pseudo[idx]
                    pred_dict['location'] = box_for_update[:3].reshape(-1,3)
                    pred_dict['dimension'] = box_for_update[3:6].reshape(-1,3)
                    pred_dict['yaw'] = box_for_update[6].reshape(-1,1)
                
                if cfg.dist or cfg.is_dp: 
                    loss_dict, loss, iou3d_histo, loss_box = model.module.get_loss(pred_dict, data, rank)   
                else:
                    if self.evi_uncertainty:
                        loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, evi_unc_addl, evi_unc_addl_epistemic, v_addl, alpha_addl, beta_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank)  
                    elif self.ensemble:
                        loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes, var = model.get_loss(pred_dict, data, rank) 
                    elif self.mcdo:
                        loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank)   
                    else:        
                        loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank)  
                
                if cfg.dataset=='KITTI' or cfg.dataset=='NUSCENES':   
                    # Get the label as what is printed on the txt file
                    if lbl=="train":
                        label, frames = format_kitti_labels(pred_dict, data, with_score=False)
                    elif lbl=="val":
                        label, frames = format_kitti_labels(pred_dict, data, with_score=(validation_loader.dataset.cfg.split=='val'))

                    # Fixing small c in 'car'
                    for idx, lab in enumerate(label):
                        if 'car' in lab:
                            label[idx] = 'C' + lab[1:]
                    
                    if nusc:                
                        boxes_lidar_nusc = get_boxes_lidar_nuscenes_format(label, frames, lbl) if nusc else None
                    else:
                        boxes_lidar_nusc = None
                    if self.debug: import pdb; pdb.set_trace()

                    # Generate pseudolabels
                    if cfg.gen_label:     
                        # Define annos variable
                        annos = get_annos_dict(label, frames, pred_dict, nusc, id_to_token, id_to_lidar_path, boxes_lidar_nusc)
                        
                        # Case for NUSC dataset
                        if nusc:
                            if len(det_annos) == 0 or data.frames[0] != det_annos[-1]["frame_id_kitti_ver"]:
                                det_annos.append(annos)                     
                            else: 
                                # No need to include additional element to det_annos (just append to the last element)
                                det_annos[-1]['name'] = np.concatenate((det_annos[-1]["name"], annos['name']), axis=0)
                                det_annos[-1]['score'] = np.concatenate((det_annos[-1]["score"], annos['score']), axis=0)
                                det_annos[-1]['boxes_lidar'] = np.concatenate((det_annos[-1]["boxes_lidar"], annos['boxes_lidar']), axis=0) 
                                det_annos[-1]['pred_labels'] = np.concatenate((det_annos[-1]["pred_labels"], annos['pred_labels']), axis=0)
                                
                            while det_annos[-1]["frame_id_kitti_ver"] != id_list[len(det_annos)-1]:
                                if cfg.gen_label:
                                    if not path.exists(f'{cfg.label_dir}'):
                                        makedirs(f'{cfg.label_dir}')
                                    file_path = path.join(f'{cfg.label_dir}', f'{id_list[len(det_annos)-1]}.txt')
                                    # if path.isfile(file_path):
                                    #     os.remove(file_path)
                                    with open(file_path, 'a') as f:
                                        #l = label[i]
                                        # score = float(l.split(' ')[-1])       # [optional]: discard 3D predictions with low confidence
                                        # if score<0.05:
                                        #     continue
                                        f.write('')
                                    
                                num_samples = 0
                                no_content_id = id_list[len(det_annos)-1]
                                det_annos.append({
                                    'name': np.zeros(num_samples), 'score': np.zeros(num_samples), 
                                    'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples),
                                    'frame_id_kitti_ver': no_content_id, 'frame_id': id_to_lidar_path[no_content_id],
                                    'metadata': {'token': id_to_token[no_content_id]}
                                })
                                det_annos[-1], det_annos[-2] = det_annos[-2], det_annos[-1]

                        # Case for KITTI dataset
                        else:
                            if len(det_annos) == 0 or data.frames[0] != det_annos[-1]["frame_id"]:
                                det_annos.append(annos)                     
                            else: 
                                # No need to include additional element to det_annos (just append to the last element)
                                det_annos[-1]['name'] = np.concatenate((det_annos[-1]["name"], annos['name']), axis=0)
                                det_annos[-1]['truncated'] = np.concatenate((det_annos[-1]["truncated"], annos['truncated']), axis=0)  
                                det_annos[-1]['occluded'] =  np.concatenate((det_annos[-1]["occluded"], annos['occluded']), axis=0) 
                                det_annos[-1]['alpha'] = np.concatenate((det_annos[-1]["alpha"], annos['alpha']), axis=0)  
                                det_annos[-1]['bbox'] = np.concatenate((det_annos[-1]["bbox"], annos['bbox']), axis=0) 
                                det_annos[-1]['dimensions'] = np.concatenate((det_annos[-1]["dimensions"], annos['dimensions']), axis=0) 
                                det_annos[-1]['location'] = np.concatenate((det_annos[-1]["location"], annos['location']), axis=0) 
                                det_annos[-1]['rotation_y'] = np.concatenate((det_annos[-1]["rotation_y"], annos['rotation_y']), axis=0) 
                                det_annos[-1]['score'] = np.concatenate((det_annos[-1]["score"], annos['score']), axis=0) 
                                det_annos[-1]['boxes_lidar'] = np.concatenate((det_annos[-1]["boxes_lidar"], annos['boxes_lidar']), axis=0)
                        
                            while det_annos[-1]["frame_id"] != id_list[len(det_annos)-1]:
                                if cfg.gen_label:
                                    if not path.exists(f'{cfg.label_dir}'):
                                        makedirs(f'{cfg.label_dir}')
                                    file_path = path.join(f'{cfg.label_dir}', f'{id_list[len(det_annos)-1]}.txt')
                                    with open(file_path, 'a') as f:
                                        f.write('')  
                                    
                                num_samples = 0
                                det_annos.append({
                                    'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                                    'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                                    'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                                    'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                                    'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
                                    'frame_id': id_list[len(det_annos)-1]
                                })
                                det_annos[-1], det_annos[-2] = det_annos[-2], det_annos[-1]
                        
                    ################################################################################################################################################ 
                    if cfg.gen_label:
                        if not path.exists(f'{cfg.label_dir}'):
                            makedirs(f'{cfg.label_dir}')
                        for i, fr in enumerate(frames):
                            file_path = path.join(f'{cfg.label_dir}', f'{fr}.txt')
                            with open(file_path, 'a') as f:
                                l = label[i]
                                # score = float(l.split(' ')[-1])       # [optional]: discard 3D predictions with low confidence
                                # if score<0.05:
                                #     continue
                                f.write(l+'\n')  
        
                # Statistics
                counter.update(loss_dict)
                histo_counter.update(iou3d_histo)

                # Append values
                if self.evi_uncertainty:
                    iou3d = np.append(iou3d, iou3d_addl)
                    evi_unc = np.append(evi_unc, evi_unc_addl)
                    evi_unc_epis = np.append(evi_unc_epis, evi_unc_addl_epistemic)
                    try: v = np.concatenate((v, v_addl), axis=0)
                    except: v = v_addl
                    try: alpha = np.concatenate((alpha, alpha_addl), axis=0)
                    except: alpha = alpha_addl
                    try: beta = np.concatenate((beta, beta_addl), axis=0)
                    except: beta = beta_addl
                    try: pred_boxes_all = np.concatenate((pred_boxes_all, pred_boxes), axis=0)
                    except: pred_boxes_all = pred_boxes
                    try: gt_boxes_all = np.concatenate((gt_boxes_all, gt_boxes), axis=0)
                    except: gt_boxes_all = gt_boxes
                elif self.ensemble:
                    iou3d = np.append(iou3d, iou3d_addl)
                    try: pred_boxes_all = np.concatenate((pred_boxes_all, pred_boxes), axis=0)
                    except: pred_boxes_all = pred_boxes
                    try: gt_boxes_all = np.concatenate((gt_boxes_all, gt_boxes), axis=0)
                    except: gt_boxes_all = gt_boxes
                    try: var_all = np.concatenate((var_all, var), axis=0)
                    except: var_all = var
                else:
                    iou3d = np.append(iou3d, iou3d_addl)
                    try: pred_boxes_all = np.concatenate((pred_boxes_all, pred_boxes), axis=0)
                    except: pred_boxes_all = pred_boxes
                    try: gt_boxes_all = np.concatenate((gt_boxes_all, gt_boxes), axis=0)
                    except: gt_boxes_all = gt_boxes
                
                pbar_text = self.get_pbar_text(counter, f'Eval-{epoch}', self.prog_metric_dir, gen_label_prints)
                process_bar.set_description(pbar_text)

            # Save for Deep Ensemble and MC Dropout
            if (self.ensemble or self.mcdo) and cfg.gen_label and self.gen_pseudo_from_external:
                import pdb; pdb.set_trace()
                pass
                # This piece of code is for SAVING FILES for each ensemble
                # with open("/home/hpaat/my_exp/MTrans-evidential/output/Ensemble/v24/pred_boxes_all.pkl", "wb") as f: 
                #     pickle.dump(pred_boxes_all, f)
                # with open("/home/hpaat/my_exp/MTrans-evidential/output/Ensemble/v24/var_all.pkl", "wb") as f: 
                #     pickle.dump(var_all, f)
                # epoch_save_path = self.conf_dir + 'conf_' + str(epoch) + '_val.pkl'         
                # epoch_save_path += '_genlabel_' + lbl  + '.pkl'
                # with open(epoch_save_path, 'wb') as file: 
                #     pickle.dump({'pred': pred_boxes_all, 'target': gt_boxes_all, 'conf': None, 'error': pred_boxes_all-gt_boxes_all, 'alea': None, 'beta': None, 'alpha': None, 'v': None, 'iou3d': iou3d, 'var': var_all}, file)
                
                # ENSEMBLE/MCDO: Get the variance from the externally saved file for FINAL result
                # with open("/home/hpaat/my_exp/MTrans-evidential/output/MCDO/mcdo_var_d1-5.pkl", 'rb') as file: 
                #     var_external = pickle.load(file)
                # with open("/home/hpaat/my_exp/MTrans-evidential/output/MCDO/conf_train_d1-5.pkl", 'wb') as file: 
                #     pickle.dump({'pred': pred_boxes_all, 'target': gt_boxes_all, 'conf': None, 'error': pred_boxes_all-gt_boxes_all, 'alea': None, 'beta': None, 'alpha': None, 'v': None, 'iou3d': iou3d, 'var': var_external}, file)
            
            # Save anything related to evidential parameters
            if self.evi_uncertainty:
                alea = beta / (alpha - 1)
                epis = beta / (v * (alpha - 1))
                alea_mean =  np.mean(alea, axis=0)
                epis_mean = np.mean(epis, axis=0)
                v_mean = np.mean(v, axis=0)
                alpha_mean = np.mean(alpha, axis=0)
                beta_mean = np.mean(beta, axis=0)

                confidence = np.sqrt(1. / ((alpha-1) * v))      # sqrt of the inverse evidence
                conf_mean = confidence.mean(axis=0)

                gt_std = gt_boxes_all.std(axis=0)
                pred_std = pred_boxes_all.std(axis=0)
                gt_mean = gt_boxes_all.mean(axis=0)
                pred_mean = pred_boxes_all.mean(axis=0)
                res_mean = (gt_boxes_all-pred_boxes_all).mean(axis=0)
                res_std = (gt_boxes_all-pred_boxes_all).std(axis=0)

                dim = ['x', 'y', 'z', 'l', 'w', 'h', 'rot']

                if gen_label_prints:
                    prog_metric_dir_corr = self.prog_metric_dir_corr + '_genlabel.txt'
                else:
                    prog_metric_dir_corr = self.prog_metric_dir_corr

                with open(prog_metric_dir_corr, "a") as file:
                    file.write(f"Eval-{epoch} {str(scipy.stats.pearsonr(iou3d, evi_unc)[0])}, evi_unc_alea_spear:{str(scipy.stats.spearmanr(iou3d, evi_unc)[0])}, ")
                    for i in range(7):
                        file.write(f"alea_corr_iou_{dim[i]}:{str(scipy.stats.pearsonr(iou3d, alea[:,i])[0])}, ")

                    # confidence
                    for i in range(7):
                        file.write(f"conf_{dim[i]}:{str(conf_mean[i])}, ")
                    for i in range(7):
                        file.write(f"conf_iou_corr_{dim[i]}:{str(scipy.stats.pearsonr(iou3d, confidence[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"res_unc_corr_{dim[i]}:{str(scipy.stats.pearsonr(alea[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"res_epis_corr_{dim[i]}:{str(scipy.stats.pearsonr(epis[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"res_conf_corr_{dim[i]}:{str(scipy.stats.pearsonr(confidence[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"v_{dim[i]}:{str(v_mean[i])}, ")
                    for i in range(7):
                        file.write(f"alpha_{dim[i]}:{str(alpha_mean[i])}, ")
                    for i in range(7):
                        file.write(f"beta_{dim[i]}:{str(beta_mean[i])}, ")
                    for i in range(7):
                        file.write(f"alea_{dim[i]}:{str(alea_mean[i])}, ")
                    for i in range(7):
                        file.write(f"epis_{dim[i]}:{str(epis_mean[i])}, ")
                    for i in range(7):
                        file.write(f"gt_std_{dim[i]}:{str(gt_std[i])}, ")
                    for i in range(7):
                        file.write(f"pred_std_{dim[i]}:{str(pred_std[i])}, ")
                    for i in range(7):
                        file.write(f"gt_mean_{dim[i]}:{str(gt_mean[i])}, ")
                    for i in range(7):
                        file.write(f"pred_mean_{dim[i]}:{str(pred_mean[i])}, ")
                    for i in range(7):
                        file.write(f"res_mean_{dim[i]}:{str(res_mean[i])}, ")
                    for i in range(7):
                        file.write(f"res_std_{dim[i]}:{str(res_std[i])}, ")
                    file.write("\n")
                with open(prog_metric_dir_corr + 'epis.txt', "a") as file:
                    file.write(f"Eval-{epoch} evi_unc_epis:{str(scipy.stats.pearsonr(iou3d, evi_unc_epis)[0])}, evi_unc_epis_spear:{str(scipy.stats.spearmanr(iou3d, evi_unc_epis)[0])}, ")
                    for i in range(7):
                        file.write(f"epis_corr_iou_{dim[i]}:{str(scipy.stats.pearsonr(iou3d, epis[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"epis_corr_iou_spear_{dim[i]}:{str(scipy.stats.spearmanr(iou3d, epis[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"alea_corr_iou_spear_{dim[i]}:{str(scipy.stats.spearmanr(iou3d, alea[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"conf_iou_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(iou3d, confidence[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"res_unc_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(alea[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"res_epis_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(epis[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                    for i in range(7):
                        file.write(f"res_conf_corr_spear_{dim[i]}:{str(scipy.stats.spearmanr(confidence[:,i], np.sqrt((gt_boxes_all-pred_boxes_all)**2)[:,i])[0])}, ")
                    file.write("\n")
                
                # Show in terminal
                print("Pearson evi all: " + str(scipy.stats.pearsonr(iou3d, evi_unc)[0]))
                print("Pearson epis all: " + str(scipy.stats.pearsonr(iou3d, evi_unc_epis)[0]))
                print("Spearman evi all: " + str(scipy.stats.spearmanr(iou3d, evi_unc)[0]))
                print("Spearman epis all: " + str(scipy.stats.spearmanr(iou3d, evi_unc_epis)[0]))

                # Save confidence, pred, target 
                if (epoch+1) % self.conf_save_interval == 0:     
                    epoch_save_path = self.conf_dir + 'conf_' + str(epoch) + '_val.pkl'         
                    if gen_label_prints: 
                        epoch_save_path += '_genlabel_' + lbl  + '.pkl'
                    if path.isfile(epoch_save_path):
                         os.remove(epoch_save_path)
                    with open(epoch_save_path, 'wb') as file:
                        pickle.dump({'pred': pred_boxes_all, 'target': gt_boxes_all, 'conf': confidence, 'error': pred_boxes_all-gt_boxes_all, 'alea':alea, 'beta':beta, 'alpha':alpha, 'v':v, 'iou3d': iou3d}, file)
        
            # If the nuscenes final frames have no content
            if nusc and cfg.gen_label:
                if len(det_annos) != len(id_list):
                    last_id_with_content = int(det_annos[-1]['frame_id_kitti_ver']) # 6015
                    while last_id_with_content < len(id_list)-1:
                        last_id_with_content = last_id_with_content+1
                        no_content_id = f"{last_id_with_content:06d}"
                        det_annos.append({
                                'name': np.zeros(num_samples), 'score': np.zeros(num_samples), 
                                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples),
                                'frame_id_kitti_ver': no_content_id, 'frame_id': id_to_lidar_path[no_content_id],
                                'metadata': {'token': id_to_token[no_content_id]}
                            })
            
            # Save the det_annos for external evaluation in pcdet
            if self.save_det_annos and cfg.gen_label:
                with open(self.home_path + '/output/' + cfg.experiment_name + '/det_annos_' + str(epoch) + '_' + lbl + '.pkl', 'wb') as f: 
                    pickle.dump(det_annos, f)

            stats = counter.average(None, group_by_description=True)
            for group in stats.keys():
                writer.add_scalars(f'Eval/{group}', stats[group], epoch)
            writer.add_histogram('Eval/iou_distribution', histo_counter.get_values(), epoch)
            
            # Metric for saving best checkpoint
            score = (counter.average(['iou3d'])['iou3d'])

        return score
    
    def predict(self, cfg,
                loader_builder,
                model,
                unlabeled_training_set,
                loader_cfg,
                counter,
                histo_counter,
                rank,
                num_gpus,
                writer,
                optim,
                scheduler,
                uncertainty_type = 'aleatoric'):
        
        # NOTE Make sure unlabeled_training_set is not shuffled (set labeled=False)
        unlabeled_training_loader = loader_builder(unlabeled_training_set, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus, labeled=False)     # Set "labeled" to False to disable shuffling for prediction

        model.eval()
        process_bar = tqdm(unlabeled_training_loader, desc='Predict', position=0, leave=True)
        counter.reset()
        histo_counter.reset()
        uncertaintys = torch.Tensor([])

        iou3d, pred_boxes_all, gt_boxes_all = np.array([]), np.array([]), np.array([])

        with torch.no_grad():
            for idx, data in enumerate(process_bar):
                data = move_to_cuda(data, 'cuda', rank)
                
                pred_dict = model(data)

                loss_dict, loss, iou3d_histo, loss_box, iou3d_addl, gt_boxes, pred_boxes = model.get_loss(pred_dict, data, rank)

                iou3d = np.append(iou3d, iou3d_addl)
                try: pred_boxes_all = np.concatenate((pred_boxes_all, pred_boxes), axis=0)
                except: pred_boxes_all = pred_boxes
                try: gt_boxes_all = np.concatenate((gt_boxes_all, gt_boxes), axis=0)
                except: gt_boxes_all = gt_boxes
                
                if uncertainty_type == 'aleatoric':
                    var = get_pred_evidential_aleatoric(pred_dict['box_uncertainty'])
                elif uncertainty_type == 'epistemic':
                    var = get_pred_evidential_epistemic(pred_dict['box_uncertainty'])
                elif uncertainty_type == 'conf':
                    var = pred_dict['conf'].view(-1)
                elif uncertainty_type == 'gt_iou':
                    var = self.get_gt_iou(pred_dict, data)
                elif uncertainty_type == 'lapl_unc':
                    var =  pred_dict['lapl_unc'].view(-1)
                uncertaintys = torch.cat([uncertaintys, var.detach().cpu()], dim=0)

                label, _ = format_kitti_labels(pred_dict, data, with_score=False)
      
                # Update unlabeled_training_set labels
                unlabeled_training_set = self.update_unlabeled_training_set(label, data, idx, unlabeled_training_set) 

                # COMMENT THIS OUT
                # if not path.exists(f'{cfg.label_dir}'):
                #     makedirs(f'{cfg.label_dir}')
                # for i, fr in enumerate(data['frames']):
                #     with open(path.join(f'{cfg.label_dir}', f'{fr}.txt'), 'a') as f:
                #         l = label[i]
                #         f.write(l+'\n')

        import pdb; pdb.set_trace()
        # Save the confidence
        with open(self.conf_dir + 'conf_not_evi.pkl', 'wb') as file:
            pickle.dump({'pred': pred_boxes_all, 'target': gt_boxes_all, 'error': pred_boxes_all-gt_boxes_all, 'iou3d': iou3d, 'pred_iou': (1/uncertaintys).detach().cpu().numpy()}, file)

        # COMMENT THIS OUT
        # train_ids = '/home/hpaat/pcdet/data/kitti/ImageSets/train.txt'
        # with open(train_ids) as f:
        #     lines = f.readlines()
        #     lines = [x.strip() for x in lines]
        # gen_id = sorted(os.listdir("/home/hpaat/my_exp/MTrans-U/pseudo_label"))
        # for gt_idx in lines:
        #     seq = gt_idx + ".txt"
        #     if seq not in gen_id:
        #         with open(path.join(f'{cfg.label_dir}', seq), 'a') as f:
        #                 f.write('')

        return unlabeled_training_set, uncertaintys
    
    def update_unlabeled_training_set(self, label, data, loader_idx, unlabeled_training_set):

        batch_size = data['images'].shape[0]
        for idx, lab in enumerate(label):
            object_label = read_label(lab, data['calibs'][idx])
            idx_in_entire_set = loader_idx * batch_size + idx
            unlabeled_training_set.update_label(idx_in_entire_set, object_label['dimensions'], object_label['location'], object_label['yaw'])
            # TODO Do we update the overlap_boxes, overlap_mask?

        return unlabeled_training_set

    def get_gt_iou(self, pred_dict, data):
        
        label, _ = format_kitti_labels(pred_dict, data, with_score=False)
        record_ious = []

        for idx, lab in enumerate(label):
            object_label = read_label(lab, data['calibs'][idx])
            boxes_lidar = np.concatenate([object_label['location'].reshape(-1,3), object_label['dimensions'].reshape(-1,3), np.array(object_label['yaw']).reshape(-1,1)], axis=1)
            try: pl_boxes_lidar = np.concatenate([pl_boxes_lidar, boxes_lidar], axis=0)
            except: pl_boxes_lidar = boxes_lidar

        gt_path = '/home/hpaat/KITTI/data_object_label_2/training/label_2/'
        
        # Get ground truth object list for this sequence
        for idx, seq in enumerate(data['frames']):
            gt_labels = gt_path + seq + '.txt'
            with open(gt_labels) as f:
                gt_lines = [x.strip() for x in f.readlines() if x[:3] =='Car']

            for _, lab in enumerate(gt_lines):
                try: object_label = read_label(lab, data['calibs'][idx])
                except: continue
                boxes_lidar = np.concatenate([object_label['location'].reshape(-1,3), object_label['dimensions'].reshape(-1,3), np.array(object_label['yaw']).reshape(-1,1)], axis=1)
                try: gt_boxes_lidar = np.concatenate([gt_boxes_lidar, boxes_lidar], axis=0)
                except: gt_boxes_lidar = boxes_lidar

            if len(gt_boxes_lidar) == 0:
                record_ious.append(float(0))
            
            pl_boxes_lidar_row_repeated = np.repeat(pl_boxes_lidar[idx].reshape(-1,7), repeats=len(gt_boxes_lidar), axis=0)
            # Get the 3D IoU of pseudolabel objects and 3D GT objects
            iou = cal_iou_3d(torch.from_numpy(pl_boxes_lidar_row_repeated).reshape(1,-1,7).float().cuda(), torch.from_numpy(gt_boxes_lidar).reshape(1,-1,7).float().cuda())
            
            record_ious.append(iou.max(1)[0].item())

        record_ious = [float(1/(iou+1e-5)) for iou in record_ious]
        return torch.tensor(record_ious)


            


