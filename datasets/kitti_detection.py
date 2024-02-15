"""
The customized Kitti Detection dataset for MAPGen. 
Each returned item will be an object with the features:
    1. object cropped image
    2. object frustum point cloud
    3. object 3D labels
    4. object frustum corresponding 2D coords
    5. foreground segmentation labels
"""

from utils.point_ops import check_points_in_box, build_image_location_map_single
from utils.calibrator import KittiCalibrator_detect
from utils.os_utils import verify_and_create_outdir
from torch.utils.data import Dataset
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from os import path
import numpy as np
import torchvision
import random
import pickle
import torch
import copy
import os

class KittiDetectionDataset(Dataset):
    def __init__(self, data_root, cfg, nusc, labeled=False, gen_pseudolabel=False, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.nusc = nusc
        if self.nusc == True:
            data_root = './data/nuscenes'
            self.root = data_root
            self.classes = [cls_name.lower() for cls_name in cfg.classes]
        else:
            self.root = data_root
            self.classes = cfg.classes
        self.pc_aug_btcdet = cfg.pc_aug_btcdet
        split = cfg.split
        if split in ['train', 'val']:
            if self.nusc == True and split in ['train']:
                split_folder = 'training'
            elif self.nusc == True and split in ['val']:
                split_folder = 'val'
            else:
                split_folder = 'training'
        elif split == 'test':
            split_folder = 'testing'
        split_root = path.join(self.root, split_folder)
        self.split_root = split_root
        
        gt_set_path = path.join(data_root, 'gt_base', split, f'gt_set_{split}.pkl')
        if path.exists(gt_set_path):
            objects = pickle.load(open(gt_set_path, 'rb'))
        else:
            verify_and_create_outdir(path.join(data_root, 'gt_base', split))
            objects = self.build_dataset()
            pickle.dump(objects, open(gt_set_path, 'wb'))
        
        # Filter objects by class and point cloud size
        self.objects = [o for o in objects if o['class'] in self.classes  \
                            and o['sub_cloud'].shape[0] >= cfg.min_points \
                            and o['foreground_label'].sum() >= (5 if split!='test' else 0)]
        # import pdb; pdb.set_trace() 
        random.seed(666)
        # Use partial frames
        labeled_frames = np.unique([o['frame'] for o in objects])
        if cfg.split == 'train' and not gen_pseudolabel:
            labeled_count = cfg.labeled_cnt
            labeled_frames_new = labeled_frames[:labeled_count] #random.sample(labeled_frames.tolist(), labeled_count)                     
            if labeled:
                labeled_frames = labeled_frames_new
            else:
                labeled_frames = list(set(labeled_frames) - set(labeled_frames_new))
            labeled_frames = np.array(labeled_frames)

        if 'partial_frames' in cfg.keys():
            self.labeled_frames = labeled_frames[:cfg.partial_frames] if cfg.partial_frames>0 else labeled_frames           # P     # Frames to be labeled # change commment: BEING DONE IN MTRANS # code changed by Helbert PAAT
            if cfg.get('use_3d_label', True):
                self.objects = [o for o in self.objects if o['frame'] in self.labeled_frames]                               # TODO  These few labeled objects?
            else:
                self.objects = [o for o in self.objects if o['frame'] not in self.labeled_frames]
        
        # Build gaussian distribution for random sampling
        self.gaussian = torch.ones(cfg.out_img_size, cfg.out_img_size) # O x O
        self.gaussian = self.gaussian / self.gaussian.sum() # O x O
        self.img_coords = build_image_location_map_single(cfg.out_img_size, cfg.out_img_size, 'cpu')    # O x O x 2         # CHANGE
        
    def __len__(self):
        return len(self.objects)

    def read_image(self, img_path):
        assert path.exists(img_path), f'{img_path} not exist'
        img = Image.open(img_path)
        img = torchvision.transforms.ToTensor()(img)    # (C, H, W)
        return img

    def read_point_cloud(self, pc_path):
        assert path.exists(pc_path), f'{pc_path} not exists'
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        return pc

    def read_calib(self, calib_path) -> KittiCalibrator_detect:
        return KittiCalibrator_detect(calib_path)

    def read_label(self, label_path, calib):
        with open(label_path) as f:
            labels = [l.strip().split(' ') for l in f.readlines()]
        object_labels = []
        btcdet_mask = []
        for label in labels:
            if label[0] == 'DontCare':
                continue
            if label[0] not in self.classes:
                continue
            cls = label[0]
            truncated = float(label[1])
            if truncated > 0.95:    # remove too much truncated
                btcdet_mask.append(False)
                continue
            btcdet_mask.append(True)

            occluded = int(label[2])
            box_2d = np.array(label[4:8], dtype=np.float32)
            dim = np.array(label[8:11], dtype=np.float32)
            loc = np.array(label[11:14], dtype=np.float32)
            yaw = float(label[14])

            # change label coordinate system: camera sys -> lidar sys
            location = calib.rect_to_velo(loc[np.newaxis, ...])
            x, y, z = location[0]
            h, w, l = dim
            z += h/2
            yaw = -yaw - np.pi/2

            object_labels.append({
                'class': cls,
                'truncated': truncated,
                'occluded': occluded,
                'box_2d': box_2d,
                'dimensions': np.array([l, w, h]),
                'location': np.array([x, y, z]),
                'yaw': yaw,
            })

            if len(label)==16:
                score = float(label[15])
                object_labels[-1]['score'] = score
        return object_labels, btcdet_mask

    def read_test_rgb_detections(self, file):
        all_labels = {}
        with open(file) as f:
            lines = [l.strip() for l in f.readlines()]
            for l in lines:
                l = l.split(' ')
                cls = l[1]
                if cls not in self.classes:
                    continue
                frame = l[0]
                if frame not in all_labels.keys():
                    all_labels[frame] = []
                truncated=-1
                occluded=-1
                box_2d = np.array(l[3:7], dtype=np.float32)
                yaw = 0
                score = float(l[2])
                all_labels[frame].append({
                    'class': cls,
                    'truncated': truncated,
                    'occluded': occluded,
                    'box_2d': box_2d,
                    'dimensions': np.array([4,1.6,1.5]),
                    'location': np.array([0,0,0]),
                    'yaw': yaw,
                    'score': score
                })
        return all_labels

    def build_overlap_matrix(self, object_labels):
        num_labels = len(object_labels)
        overlap_matrix = [[] for _ in range(num_labels)]
        for i in range(num_labels):
            for j in range(i+1, num_labels):
                b1 = object_labels[i]['box_2d']     # left, top, right, bottom
                b2 = object_labels[j]['box_2d']
                overlap_vertical = max(b1[1], b2[1]) < min(b1[3], b2[3])
                overlap_horizontal = max(b1[0], b2[0]) < min(b1[2], b2[2]) 
                if overlap_vertical and overlap_horizontal:
                    overlap_matrix[i].append(j)
                    overlap_matrix[j].append(i)
        return overlap_matrix

    def build_dataset(self):
        
        print("========== Building Dataset ==========")
        all_objects = []
        split_root = self.split_root
        split = self.cfg.split
        out_3d_dir = path.join(self.root, 'processed', 'points_3d', split)
        out_2d_dir = path.join(self.root, 'processed', 'points_2d', split)

        try:
            verify_and_create_outdir(out_2d_dir)
            verify_and_create_outdir(out_3d_dir)
        except:
            print("[Warning] Preprocessed PointClouds and Images already exists.")

        if self.nusc:
            all_frames = []
            if split == 'train':
                for id in range(28130):   #28130
                    all_frames.append(f"{id:06d}")
            elif split == 'val':
                for id in range(6019):
                    all_frames.append(f"{id:06d}")
        else:
            with open(path.join(self.root, 'ImageSets', f'{split}.txt')) as f:
                all_frames = [l.strip() for l in f.readlines()]

        if self.cfg.split == 'test':
            test_all_labels = self.read_test_rgb_detections(self.cfg.test_rgb_file)
            
        for frame in tqdm(all_frames, desc=f"Processing {split} data"):
            # Preprocess frames, taking out the points within image scope, and their projected 2D coords
            img = self.read_image(path.join(split_root, 'image_2', f'{frame}.png'))
            H, W = img.shape[1:3]
            point_cloud = self.read_point_cloud(path.join(split_root, 'velodyne', f'{frame}.bin'))
            calib = self.read_calib(path.join(split_root, 'calib', f'{frame}.txt'))
            
            ############################################################################################################################################
            # Include the point cloud completion from the BtcDet            # Modified-Added-Changed by HP
            if self.pc_aug_btcdet:
                object_labels, btcdet_mask = self.read_label(path.join(split_root, 'label_2', f'{frame}.txt'), calib)

                # If the frame is in the list of IDs, include the additional points
                id_addpoints_list_all = os.listdir(r"/home/hpaat/my_exp/BtcDet/data/kitti/detection3d/bm_50maxdist_2num_Car")
                start = str(int(frame)) + "_"
                id_addpoints_frame = sorted([filename for filename in id_addpoints_list_all if filename.startswith(start)])
                id_addpoints_frame = np.array(id_addpoints_frame)[btcdet_mask]
                if len(id_addpoints_frame) > 0:
                    for index, file in enumerate(id_addpoints_frame):
                        with open("/home/hpaat/my_exp/BtcDet/data/kitti/detection3d/bm_50maxdist_2num_Car/" + file, "rb") as f:
                            point_cloud_addl = pickle.load(f)
                        try: point_cloud_addl += object_labels[index]["location"]
                        except: 
                            print(frame, "\n")
                            print(object_labels, "\n")
                            print(id_addpoints_frame, "\n")
                        point_cloud_addl = np.hstack((point_cloud_addl, np.zeros([point_cloud_addl.shape[0],1], point_cloud_addl.dtype)))
                        point_cloud = np.vstack((point_cloud, point_cloud_addl))
            ############################################################################################################################################
            
            # Determine which falls inside the image and has depth>=0
            p2d_float, depth = calib.velo_to_cam(point_cloud[:, :3])
            x, y = p2d_float[:, 0], p2d_float[:, 1]
            idx = np.logical_and.reduce([depth>=0, x>=0, x<W, y>=0, y<H])
            point_cloud = point_cloud[idx]
            p2d_float = p2d_float[idx]
            point_cloud.astype(np.float32).tofile(path.join(out_3d_dir, f'{frame}.bin'))
            p2d_float.astype(np.float32).tofile(path.join(out_2d_dir, f'{frame}.bin'))

            # Build object-level dataset
            if self.cfg.split != 'test':
                object_labels, _ = self.read_label(path.join(split_root, 'label_2', f'{frame}.txt'), calib)
            else:
                if frame in test_all_labels.keys():
                    object_labels = test_all_labels[frame]
                else:
                    continue
            overlap_matrix = self.build_overlap_matrix(object_labels)                   # TODO Understand this variable
            
            for i, obj in enumerate(object_labels):
                obj['frame'] = frame

                # query sub cloud within the 2D box
                left, top, right, bottom = obj['box_2d']
                idx = np.logical_and.reduce([p2d_float[:, 0]>left, p2d_float[:, 1]>top, p2d_float[:, 0]<right, p2d_float[:, 1]<bottom])
                sub_cloud = point_cloud[idx]
                sub_cloud2d = p2d_float[idx]
                obj['sub_cloud'] = sub_cloud
                obj['sub_cloud2d'] = sub_cloud2d

                # Generate foreground label     # NOTE This is only used for the loss calculation
                foreground_label = check_points_in_box(sub_cloud[:, :3], location=obj['location'], dimension=obj['dimensions'], yaw=obj['yaw']) 
                obj['foreground_label'] = foreground_label

                overlap_boxes = [object_labels[j]['box_2d'] for j in overlap_matrix[i]]
                obj['overlap_boxes'] = overlap_boxes

                # Modified-Added-Changed by Helbert PAAT
                obj['weight'] = float(1)

                all_objects.append(obj)
        
        return all_objects

    def __getitem__(self, index):
        obj = copy.deepcopy(self.objects[index])
        obj = self.load_object_full_data(obj)
        return obj
    
    def update_label(self, index, new_dim, new_loc, new_yaw):
        self.objects[index]['dimensions'] = new_dim
        self.objects[index]['location'] = new_loc
        self.objects[index]['yaw'] = new_yaw 

    def update_weights(self, loss_weights):
        for idx in range(len(self.objects)):
            self.objects[idx]['weight'] = loss_weights[idx]

    def load_object_full_data(self, obj):
        class_name = obj['class']
        obj['class_idx'] = self.classes.index(class_name) # TODO So this is always 0?
        obj['use_3d_label'] = self.cfg.get('use_3d_label', True)
        cloud_size = self.cfg.out_cloud_size

        split_root = self.split_root
        full_img = self.read_image(path.join(split_root, 'image_2', f"{obj['frame']}.png")) # C x H x W

        # build overlap mask # TODO What is the intuition with the overlap mask?
        overlap_mask = torch.ones_like(full_img[0:1, :, :]) # 1 x H x W
        for olb in obj['overlap_boxes']: # Empty for some indices
            l, t, r, b = olb
            l, t, r, b = int(np.floor(l)), int(np.floor(t)), int(np.ceil(r)), int(np.ceil(b))
            overlap_mask[:, t:b, l:r] = 0
        full_img = torch.cat([full_img, overlap_mask], dim=0) # C+1 x H x W

        l, t, r, b = obj['box_2d']
        # box2d augmentation, random scale + shift # TODO What is the intuition of doing this? 
        if self.cfg.get('box2d_augmentation', False):
            random_scale = np.random.rand(2) * 0.2 + 0.95        # 2 # [95% ~ 115%]
            random_shift = np.random.rand(2) * 0.1 - 0.05        # 2 # [-5% ~ 5%]
            tw, th, tx, ty = r-l, b-t, (r+l)/2, (t+b)/2                 # width, height, center x, center  y
            tx, ty = tx + tw*random_shift[0], ty + th*random_shift[1]   # random shift
            tw, th = tw * random_scale[0], th*random_scale[1]           # random scale
            l, t, r, b = max(0, tx-tw/2), max(0, ty-th/2), min(tx+tw/2, full_img.shape[2]-1), min(ty+th/2, full_img.shape[1]-1) # new l,t,r,b
        
            # re-crop frustum sub-cloud, and cloud's 2D projection
            frame = obj['frame']
            all_points = np.fromfile(path.join(self.root, 'processed', 'points_3d', self.cfg.split, f'{frame}.bin'), dtype=np.float32).reshape(-1,4)   # N x 4
            all_points2d = np.fromfile(path.join(self.root, 'processed', 'points_2d', self.cfg.split, f'{frame}.bin'), dtype=np.float32).reshape(-1,2) # N x 2
            idx = np.logical_and.reduce([all_points2d[:, 0]>l, all_points2d[:, 1]>t, all_points2d[:, 0]<r, all_points2d[:, 1]<b])   # N # TODO Intuition of using augmented 2D box?
            obj['sub_cloud'] = all_points[idx]  # Nl x 4 # Getting only 3d points whose 2d projection fall inside the shifted and scaled 2D box
            obj['sub_cloud2d'] = all_points2d[idx] # Nl x 2
            obj['foreground_label'] = check_points_in_box(obj['sub_cloud'][:, :3], location=obj['location'], dimension=obj['dimensions'], yaw=obj['yaw']) # Nl

        l, t, r, b = int(np.floor(l)), int(np.floor(t)), int(np.ceil(r)), int(np.ceil(b))
        img = full_img[:,t:b, l:r].unsqueeze(0)  # (1, 4, box_h, box_w)     # Crop the image to get the image corresponding to the 2D box only

        # crop original image by the obj's 2D box
        box_size = max(b-t, r-l)
        out_shape = self.cfg.out_img_size
        img = torch.nn.functional.interpolate(img, scale_factor=out_shape/box_size, mode='bilinear', align_corners=True, recompute_scale_factor=False) # (1, 4, new_box_h, new_box_w)
        h, w = img.shape[-2:]
        num_padding = (int(np.floor((out_shape-w)/2)), int(np.ceil((out_shape-w)/2)), int(np.floor((out_shape-h)/2)), int(np.ceil((out_shape-h)/2)))
        img = torch.nn.functional.pad(img, num_padding)     # 1 x 4 x W x W # Zero-padding to make it square
        crop_sub_cloud2d = (obj['sub_cloud2d'] - np.array([l, t])) * (out_shape/box_size) + np.array([num_padding[0], num_padding[2]]) # TODO Intuition? # Nl x 2
        try:
            assert np.logical_and.reduce([crop_sub_cloud2d[:, 0]>=0, crop_sub_cloud2d[:,0]<=112.01, crop_sub_cloud2d[:, 1]>=0, crop_sub_cloud2d[:,1]<=112.01]).all()  
        except:
            print("l=", l) 
            print("t=", t)
            print("r=", r)
            print("b=", b)
            print("box_size=", box_size)
            print("num_padding=", num_padding)
            print("box_2d=", obj['box_2d']) 
            print("max of sub_cloud2d=", np.max(obj['sub_cloud2d'], axis=0))
            print("min of sub_cloud2d=", np.min(obj['sub_cloud2d'], axis=0)) 
            import sys
            sys.exit("Error message")       

        # img, overlap_mask, pos_mask = img[:, 0:3, :, :], img[:, 3:4, :, :], img[:, 4:6, :, :]
        img, overlap_mask = img[:, 0:3, :, :], img[:, 3:4, :, :] # 1 x 3 x W x W    # 1 x 1 x W x W # TODO What is the intuition of the overlap_mask?

        # sampling the point cloud to fixed size
        out_sub_cloud = np.ones((cloud_size, 4))* (-9999)       # cloud_size x 4 # -9999 for paddings
        out_sub_cloud2d = np.ones((cloud_size, 2)) * (-9999)    # cloud_size x 4 # -9999 for paddings
        out_ori_cloud2d = np.ones((cloud_size, 2)) * (-9999)    # cloud_size x 2 
        out_real_point_mask = np.zeros((cloud_size))    # cloud_size  # 0 for padding, 1 for real points, 2 for masked, 3 for jittered
        out_foreground_label = np.ones((cloud_size))*2  # cloud_size  # 0 for background, 1 for foreground, 2 for unknown
        

        sub_cloud = obj['sub_cloud']     # Nl x 4
        sub_cloud2d = obj['sub_cloud2d'] # Nl x 2
        foreground_label = obj['foreground_label'] # Nl
        out_cloud_size = self.cfg.out_cloud_size# 512
        if sub_cloud.shape[0] > out_cloud_size:
            sample_idx = np.random.choice(np.arange(sub_cloud.shape[0]), out_cloud_size, replace=False)     # out_cloud_size=512       # random sampling
            out_sub_cloud[...] = sub_cloud[sample_idx]              # out_cloud_size x 4 # TODO Why is this the size? 
            out_sub_cloud2d[...] = crop_sub_cloud2d[sample_idx]     # out_cloud_size x 2 # TODO Understand crop_sub_cloud2d
            out_ori_cloud2d[...] = sub_cloud2d[sample_idx]          # out_cloud_size x 2
            out_real_point_mask[...] = 1                            # out_cloud_size # TODO All 1 for now?
            out_foreground_label[...] = foreground_label[sample_idx]    # out_cloud_size
        elif sub_cloud.shape[0] <= out_cloud_size:
            pc_size = sub_cloud.shape[0]
            out_sub_cloud[:pc_size] = sub_cloud             # Nl x 4
            out_sub_cloud2d[:pc_size] = crop_sub_cloud2d    # Nl x 2
            out_ori_cloud2d[:pc_size] = sub_cloud2d         # Nl x 2
            out_real_point_mask[:pc_size] = 1               # Nl
            out_foreground_label[:pc_size] = foreground_label # Nl

            # sample 2D points, leave blank for 3D coords
            p = ((img[0]!=0).all(dim=0) * 1).numpy().astype(np.float64)   # W x W # only sample pixels from not-padding-area
            p = p / p.sum() # W x W
            resample = (p>0).sum() < (out_cloud_size - pc_size) # TODO Why do the comparison?
            sample_idx = np.random.choice(np.arange(out_shape * out_shape), out_cloud_size - pc_size, replace=resample,
                                          p=p.reshape(-1)) # (out_cloud_size - pc_size)
            sampled_c2d = self.img_coords.view(-1, 2)[sample_idx, :].numpy() # (out_cloud_size - pc_size) x 2
            out_sub_cloud2d[pc_size:, :] = sampled_c2d      # TODO How about the out_sub_cloud?
            out_ori_cloud2d[pc_size:, :] = (sampled_c2d - np.array([num_padding[0], num_padding[2]])) / (out_shape/box_size) + np.array([l, t]) 
            

            assert np.logical_and.reduce([out_ori_cloud2d[:pc_size, 0]>=l, out_ori_cloud2d[:pc_size,0]<=r,
                                     out_ori_cloud2d[:pc_size, 1]>=t, out_ori_cloud2d[:pc_size,1]<=b]).all()

        # random mask/jitter points
        num_real_points = (out_real_point_mask==1).sum() 
        mask_ratio = np.random.rand() * (self.cfg.mask_ratio[1] - self.cfg.mask_ratio[0]) + self.cfg.mask_ratio[0]  # How many will we mask? # randomly choose from (r_min, r_max) # self.cfg.mask_ratio = [0.25, 0.95]
        num_mask = min(int(mask_ratio * num_real_points), max(0, num_real_points - 5)) # leave at least 5 points # How often is the case that num_real_points<=5
        idx = np.random.choice(np.arange(num_real_points), num_mask, replace=False)           # num_mask 
        mask_idx = idx                                                                        # num_mask 
        out_real_point_mask[mask_idx] = 2           # out_cloud_size   # 2 for masked   # TODO Intuition? 

        # load calib
        if self.cfg.load_calib:
            calib = self.read_calib(path.join(self.split_root, 'calib', f"{obj['frame']}.txt"))
            obj['calib'] = calib

        obj['frame_img'] = img                              # 1 x 3 x W x W
        obj['sub_cloud'] = out_sub_cloud                    # out_cloud_size x 4
        obj['sub_cloud2d'] = out_sub_cloud2d                # out_cloud_size x 2
        obj['ori_cloud2d'] = out_ori_cloud2d                # out_cloud_size x 2
        obj['real_point_mask'] = out_real_point_mask        # out_cloud_size
        obj['foreground_label'] = out_foreground_label      # out_cloud_size
        obj['overlap_mask'] = overlap_mask                  # 1 x 1 x W x W

        return obj

def stat_dataset(dataset, info):
    print(f'\n\n####### Statistics {info} #######\n\n')
    for c in dataset.classes:
        class_objs = [o for o in dataset.objects if o['class']==c]
        num_samples = len(class_objs)
        print(f'==== {c} X {num_samples} ====')
        h = [b[3]-b[1] for b in [o['box_2d'] for o in class_objs]]
        w = [b[2]-b[0] for b in [o['box_2d'] for o in class_objs]]
        print('avg 2D (H, W): ', sum(h)/len(h), sum(w)/len(w))
        print('max 2D (H, W): ', max(h), max(w))
        print('min 2D (H, W): ', min(h), min(w))
        h = [b[0] for b in [o['dimensions'] for o in class_objs]]
        w = [b[1] for b in [o['dimensions'] for o in class_objs]]
        l = [b[2] for b in [o['dimensions'] for o in class_objs]]
        print('avg 3D (H, W, L): ', sum(h)/len(h), sum(w)/len(w), sum(l)/len(l))
        print('max 3D (H, W, L): ', max(h), max(w), max(l))
        print('min 3D (H, W, L): ', min(h), min(w), min(l))
        sc = [o['sub_cloud'].shape[0] for o in class_objs]
        print('avg points: ', sum(sc)/len(sc))
        print('max points: ', max(sc))
        print('min points: ', min(sc))
        foreground_points = [o['foreground_label'].sum() for o in class_objs]
        print('avg foreground points: ', sum(foreground_points)/len(foreground_points))
        print('max foreground points: ', max(foreground_points))
        print('min foreground points: ', min(foreground_points))
        print('\n\n')

if __name__=='__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Build Kitti Detection Dataset')
    parser.add_argument('-build_dataset', action='store_true')
    parser.add_argument('--cfg_file', type=str, help='Config file')
    args = parser.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))

    if args.build_dataset:
        root = cfg.data_root
        train_config = EasyDict(cfg.DATASET_CONFIG.TRAIN_SET)
        dataset = KittiDetectionDataset(root, train_config)
        val_config = EasyDict(cfg.DATASET_CONFIG.VAL_SET)
        dataset = KittiDetectionDataset(root, val_config)
    print('========== Finish Building ==========')
    

    dataset = KittiDetectionDataset(cfg.data_root, cfg.DATASET_CONFIG.TRAIN_SET)
    stat_dataset(dataset, 'Training set')
    dataset = KittiDetectionDataset(cfg.data_root, cfg.DATASET_CONFIG.VAL_SET)
    stat_dataset(dataset, 'Validation set')