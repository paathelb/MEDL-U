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

def get_pbar_text(counter, prefix):
    #stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf', 'evidential_loss'])
    #pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, evi_loss:{stats['evidential_loss']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f}"
    
    stats = counter.average(['loss_box', 'loss_segment', 'loss_depth', 'loss_conf', 'loss_dir', 'loss', 'iou3d', 'segment_iou', 'err_dist', 'recall_7', 'acc_dir', 'err_conf', \
                             'lapl_unc_checker', 'loss_loc_lapl', 'loss_dim_lapl', 'loss_yaw_lapl'])
    pbar_text = f"{prefix} l_iou:{stats['loss_box']:.2f}, l_seg:{stats['loss_segment']:.2f}, l_depth:{stats['loss_depth']:.2f}, l_conf:{stats['loss_conf']:.2f}, l_dir:{stats['loss_dir']:.2f}, L:{stats['loss']:.2f}, Seg:{stats['segment_iou']*100:.2f}, XYZ:{stats['err_dist']:.2f}, IoU:{stats['iou3d']*100:.2f}, R:{stats['recall_7']*100:.2f}, Dr:{stats['acc_dir']*100:.2f}, Cf: {stats['err_conf']*100:.2f},    \
        lapl_unc_err: {stats['lapl_unc_checker']*100:.2f}, lapl_loc_loss: {stats['loss_loc_lapl']:.2f}, lapl_dim_loss: {stats['loss_dim_lapl']:.2f}, lapl_yaw_loss: {stats['loss_yaw_lapl']:.2f}"
    return pbar_text

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

class runner():
    def _init__(self):
        self.dummy  = None

    def run(self, loader_builder, training_set, unlabeled_training_set, validation_set, start_epoch, cfg, train_cfg, loader_cfg, temp_cfg, model, optim, scheduler, counter, histo_counter, writer, rank, num_gpus, episode_num = None, init_run=False):

        training_loader = loader_builder(training_set, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus)
        validation_loader = loader_builder(validation_set, cfg, loader_cfg.VAL_LOADER, rank, num_gpus)
        if unlabeled_training_set is not None:
            unlabeled_training_loader = loader_builder(unlabeled_training_set, cfg, loader_cfg.TRAIN_LOADER, rank, num_gpus)        # loaders must all be shuffled
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
          
            if ((actual_epoch + 1) % train_cfg.epoches_per_eval) == 0 and actual_epoch >= train_cfg.eval_begin:
                score = self.eval(cfg, model, validation_loader, counter, histo_counter, actual_epoch, writer, rank, lbl="val")         # may save det_annos & pseudolabels     
                if score > best_score:
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
                loss_dict, loss, iou3d_histo, loss_box = model.get_loss(pred_dict, data, rank, weights)
            histo_counter.update(iou3d_histo)
            
            # statistics
            counter.update(loss_dict)
            #loss_box.register_hook(lambda grad: print(grad))
            loss.backward()
            #for p in model.parameters():
            #    print(p.grad.norm())
            
            optim.step()
            scheduler.step()
            counter.update({'lr':(optim.param_groups[0]['lr'], 1, 'learning_rate')})

            pbar_text = get_pbar_text(counter, f'T-{epoch}')        
            process_bar.set_description(pbar_text)

        stats = counter.average(None, group_by_description=True)
        for group in stats.keys():
            writer.add_scalars(f'Train/{group}', stats[group], epoch)
        writer.add_histogram('Train/iou_distribution', histo_counter.get_values(), epoch)
        
    def eval(self, cfg, model, validation_loader, counter, histo_counter, epoch, writer, rank, lbl):
        # TODO Put to cfg file
        home_path =         '/home/hpaat/my_exp/MTrans-U'
        data_path =         '/data/kitti_detect'
        val_link_path =     '/home/hpaat/pcdet/data/nuscenes/kitti_ver/val/val_link.txt'
        nusc = cfg.dataset == 'NUSCENES'
        
        if nusc:
            nuscenes = NuScenes(dataroot='/home/hpaat/pcdet/data/nuscenes/v1.0-trainval', version='v1.0-trainval')
            id_to_token = {}
            with open(val_link_path) as f:
                val_link = f.readlines()
            for line in val_link:
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

            # Modified/Changed/Added by HP      # Get access to train/val IDs
            if not nusc:
                with open(home_path + data_path + '/ImageSets/' + lbl + '.txt') as f:
                    id_list = f.readlines()
                id_list = [id.strip() for id in id_list]
                id_to_lidar_path = None
            elif nusc and lbl=='val': 
                id_list = [f"{id:06d}" for id in range(6019)]
                id_to_lidar_path = {}
                for id in id_list:
                    sample_token = id_to_token[id]
                    sample = nuscenes.get('sample', sample_token)
                    lidar_token = sample['data']['LIDAR_TOP']
                    sd_record_lid = nuscenes.get('sample_data', lidar_token)
                    filename_lid_full = sd_record_lid['filename']
                    id_to_lidar_path[id] = filename_lid_full.split('/')[-1] 
            
            ######################################################################################################################################################
            
            for data in process_bar:
                data = EasyDict(data)
                data = move_to_cuda(data, 'cuda', rank)
                pred_dict = model(data)
                if cfg.dist or cfg.is_dp: 
                    loss_dict, loss, iou3d_histo, loss_box = model.module.get_loss(pred_dict, data, rank)   
                else:
                    loss_dict, loss, iou3d_histo, loss_box = model.get_loss(pred_dict, data, rank)         
                
                if cfg.dataset=='KITTI':   
                    # Get the label as what is printed on the txt file
                    if lbl=="train":
                        label, frames = format_kitti_labels(pred_dict, data, with_score=(validation_loader.dataset.cfg.split=='train'), nusc=True)
                    elif lbl=="val":
                        label, frames = format_kitti_labels(pred_dict, data, with_score=(validation_loader.dataset.cfg.split=='val'), nusc=True)
                                    
                    #boxes_lidar_nusc = get_boxes_lidar_nuscenes_format(label, frames, lbl) if nusc else None
                    boxes_lidar_nusc = None

                    # Define annos variable
                    annos = get_annos_dict(label, frames, pred_dict, nusc, id_to_token, id_to_lidar_path, boxes_lidar_nusc)
                    
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
                                with open(path.join(f'{cfg.label_dir}', f'{id_list[len(det_annos)-1]}.txt'), 'a') as f:
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
                                with open(path.join(f'{cfg.label_dir}', f'{id_list[len(det_annos)-1]}.txt'), 'a') as f:
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
                            with open(path.join(f'{cfg.label_dir}', f'{fr}.txt'), 'a') as f:
                                l = label[i]
                                # score = float(l.split(' ')[-1])       # [optional]: discard 3D predictions with low confidence
                                # if score<0.05:
                                #     continue
                                f.write(l+'\n')  
        
                # Statistics
                counter.update(loss_dict)
                histo_counter.update(iou3d_histo)

                pbar_text = get_pbar_text(counter, f'Eval')
                process_bar.set_description(pbar_text)

            # If the final frames have no content
            if nusc:
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
            save_det_annos = False       # TODO Add to config
            if save_det_annos:
                with open(home_path + '/output/' + cfg.experiment_name + '/det_annos_' + str(epoch) + '.pkl', 'wb') as f: 
                    pickle.dump(det_annos, f)

            stats = counter.average(None, group_by_description=True)
            for group in stats.keys():
                writer.add_scalars(f'Eval/{group}', stats[group], epoch)
            writer.add_histogram('Eval/iou_distribution', histo_counter.get_values(), epoch)
            
            # metric for saving best checkpoint
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

        with torch.no_grad():
            for idx, data in enumerate(process_bar):
                data = move_to_cuda(data, 'cuda', rank)
                
                pred_dict = model(data)
                
                if uncertainty_type == 'aleatoric':
                    var = get_pred_evidential_aleatoric(pred_dict['box_uncertainty'])
                elif uncertainty_type == 'epistemic':
                    var = get_pred_evidential_aleatoric(pred_dict['box_uncertainty'])
                elif uncertainty_type == 'conf':
                    var = pred_dict['conf'].view(-1)
                elif uncertainty_type == 'gt_iou':
                    var = self.get_gt_iou(pred_dict, data)
                elif uncertainty_type == 'lapl_unc':
                    var =  pred_dict['lapl_unc'].view(-1)
                uncertaintys = torch.cat([uncertaintys, var.detach().cpu()], dim=0)

                label, _ = format_kitti_labels(pred_dict, data, with_score=False, nusc=True)
      
                # Update unlabeled_training_set labels
                unlabeled_training_set = self.update_unlabeled_training_set(label, data, idx, unlabeled_training_set) 

                # COMMENT THIS OUT
                # if not path.exists(f'{cfg.label_dir}'):
                #     makedirs(f'{cfg.label_dir}')
                # for i, fr in enumerate(data['frames']):
                #     with open(path.join(f'{cfg.label_dir}', f'{fr}.txt'), 'a') as f:
                #         l = label[i]
                #         f.write(l+'\n')

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
        
        label, _ = format_kitti_labels(pred_dict, data, with_score=False, nusc=True)
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


            


