"""
The customized Nuscenes Detection dataset for MTrans. 
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
import pickle
import torch
import copy
import os
import json

from nuscenes.nuscenes import NuScenes
from utils.geo_trans_torch import GeoTransTorch
from pyquaternion import Quaternion
from torchvision.utils import save_image

class NuscenesDetectionDataset(Dataset):
    def __init__(self, data_root, cfg, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.root = data_root
        self.classes = cfg.classes
        self.pc_aug_btcdet = cfg.pc_aug_btcdet
        split = cfg.split
        if split in ['train', 'val']:
            split_folder = 'training'
        elif split == 'test':
            split_folder = 'testing'
        split_root = path.join(self.root, split_folder)
        self.split_root = split_root        # './data/nuscenes/training'

        gt_set_path = path.join(data_root, 'gt_base', split, f'gt_set_{split}.pkl')
        if path.exists(gt_set_path):
            objects = pickle.load(open(gt_set_path, 'rb'))  # object len is 14k   # dict_keys(['class', 'truncated', 'occluded', 'box_2d', 'dimensions', 'location', 'yaw', 'frame', 'sub_cloud', 'sub_cloud2d', 'foreground_label', 'overlap_boxes'])
        else:
            verify_and_create_outdir(path.join(data_root, 'gt_base', split))
            objects = self.build_dataset()
            pickle.dump(objects, open(gt_set_path, 'wb'))
    
        # Filter objects by class and point cloud size
        self.objects = [o for o in objects if o['class'] in cfg.classes \
                            and o['sub_cloud'].shape[0] >= cfg.min_points \
                            and o['foreground_label'].sum() >= (5 if split!='test' else 0)]
        
        # Use partial frames
        labeled_frames = np.unique([o['frame'] for o in objects])
        if 'partial_frames' in cfg.keys():
            self.labeled_frames = labeled_frames[:cfg.partial_frames] if cfg.partial_frames>0 else labeled_frames # P # Frames to be labeled # change commment: BEING DONE IN MTRANS # code changed by Helbert PAAT
            if cfg.get('use_3d_label', True):
                self.objects = [o for o in self.objects if o['frame'] in self.labeled_frames] # TODO These few labeled objects?
            else:
                self.objects = [o for o in self.objects if o['frame'] not in self.labeled_frames]

        # Build gaussian distribution for random sampling
        self.gaussian = torch.ones(cfg.out_img_size, cfg.out_img_size) # O x O
        self.gaussian = self.gaussian / self.gaussian.sum() # O x O
        self.img_coords = build_image_location_map_single(cfg.out_img_size, cfg.out_img_size, 'cpu') # O x O x 2 # CHANGE
        
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

    def view_points(self, points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
        """
        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.

        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
        all zeros) and normalize=False

        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
            The projection should be such that the corners are projected onto the first 2 axis.
        :param normalize: Whether to normalize the remaining coordinate (along the third axis).
        :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """
        # from ./nuscenes/utils/geometry_utils.py

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def map_custom_pointcloud_to_image(self, nusc, pointsensor_token, camera_token, box3d, min_dist=1.0, 
                                       show_lidarseg=False, show_panoptic=False, render_intensity=False):
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        if pointsensor['sensor_modality'] == 'lidar':
            if show_lidarseg or show_panoptic:
                pass
        else:
            pass
        #im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

        points = box3d.view(-1, 4).T.numpy()        # 4 x N*8
        
        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        #pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        points[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix, points[:3, :])
        #pc.translate(np.array(cs_record['translation']))
        for i in range(3):
            points[i, :] = points[i, :] + np.array(cs_record['translation'])[i]

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        #pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        points[:3, :] = np.dot(Quaternion(poserecord['rotation']).rotation_matrix, points[:3, :])
        #pc.translate(np.array(poserecord['translation']))
        for i in range(3):
            points[i, :] = points[i, :] + np.array(poserecord['translation'])[i]

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        #pc.translate(-np.array(poserecord['translation']))
        for i in range(3):
            points[i, :] = points[i, :] + -np.array(poserecord['translation'])[i]
        #pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
        points[:3, :] = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points[:3, :])

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        #pc.translate(-np.array(cs_record['translation']))
        for i in range(3):
            points[i, :] = points[i, :] + -np.array(cs_record['translation'])[i]
        #pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        points[:3, :] = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points[:3, :])

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = points[2, :]

        if render_intensity:
            pass
        elif show_lidarseg or show_panoptic:
            pass
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = self.view_points(points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        # mask = np.ones(depths.shape[0], dtype=bool)
        # mask = np.logical_and(mask, depths > min_dist)
        # mask = np.logical_and(mask, points[0, :] > 1)
        # mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        # mask = np.logical_and(mask, points[1, :] > 1)
        # mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        # points = points[:, mask]
        # coloring = coloring[mask]
     
        return points[:2,:].T.reshape(-1, 8, 2)

    def encode_box2d(self, nusc, pointsensor_token, camera_token, rotys, dims, locs, img_size, bound_corners=True):
        """
        Only support objects in a single image, because of img_size
        :param K:
        :param rotys:
        :param dims:
        :param locs:
        :param img_size: [w, h]
        :return: bboxfrom3d, shape = [N, 4]
        """
        rotys = torch.from_numpy(rotys)
        dims = torch.from_numpy(dims)
        locs = torch.from_numpy(locs)
        device = rotys.device
        #K = K.to(device=device)

        box3d = GeoTransTorch.encode_box3d(rotys, dims, locs)
        pts_3d_rect = GeoTransTorch.cart2hom(box3d)
        box3d_image = self.map_custom_pointcloud_to_image(nusc, pointsensor_token, camera_token, pts_3d_rect)
        box3d_image = torch.from_numpy(box3d_image)
        
        xmins, _ = box3d_image[:, :, 0].min(dim=1)
        xmaxs, _ = box3d_image[:, :, 0].max(dim=1)
        ymins, _ = box3d_image[:, :, 1].min(dim=1)
        ymaxs, _ = box3d_image[:, :, 1].max(dim=1)
        
        if bound_corners:
            xmins = xmins.clamp(0, img_size[0])
            xmaxs = xmaxs.clamp(0, img_size[0])
            ymins = ymins.clamp(0, img_size[1])
            ymaxs = ymaxs.clamp(0, img_size[1])

        bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                                xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)

        return bboxfrom3d.numpy() 

    def get_box2d(self, nusc, scene, image_annotations):
        sample_token_data = nusc.get('sample', scene['token'])
        sample_anns_token = sample_token_data['anns']
        
        box_2d = []
        for anns in sample_anns_token:
            box_2d.append(next(item['bbox_corners'] for item in image_annotations if item['sample_annotation_token'] == anns))

        return np.array(box_2d)

    def prep_label(self, scene_data, box_2d_list):
        object_labels, btcdet_mask = [], []

        for index, class_type in enumerate(scene_data['gt_names']):
            if str(class_type) not in [x.lower() for x in self.classes]:
                continue
            cls = str(class_type)
            truncated = 0           # TODO Determine truncated level of objects in every scene of nuscenes 
            if truncated > 0.95:    # remove too much truncated
                btcdet_mask.append(False)
                continue
            btcdet_mask.append(True)

            occluded = 0            # TODO Determine truncated level of objects in every scene of nuscenes 
            box_2d = box_2d_list[index]           # TODO Create a function to convert scene_data['gt_boxes'][index,:] to 2D  # np.array(label[4:8], dtype=np.float32)
            dim = scene_data['gt_boxes'][index,3:6] # hwl # NOTE not float32, but float64
            loc = scene_data['gt_boxes'][index,0:3] # np.array(label[11:14], dtype=np.float32)
            yaw = float(scene_data['gt_boxes'][index,7])

            # TODO Is this needed?
            # change label coordinate system: camera sys -> lidar sys
            # location = calib.rect_to_velo(loc[np.newaxis, ...])
            # x, y, z = location[0]
            # h, w, l = dim
            # z += h/2
            # yaw = -yaw - np.pi/2

            object_labels.append({
                'class': cls,
                'truncated': truncated,
                'occluded': occluded,
                'box_2d': box_2d,
                'dimensions': dim, # np.array([l, w, h]),
                'location': loc, # np.array([x, y, z]),
                'yaw': yaw,
            })

            # if len(label)==16:
            #     score = float(label[15])
            #     object_labels[-1]['score'] = score

        return object_labels, btcdet_mask

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

        if self.cfg.split=='test':
            test_all_labels = self.read_test_rgb_detections(self.cfg.test_rgb_file)

        # with open(path.join(self.root, 'ImageSets', f'{split}.txt')) as f:
        #     all_frames = [l.strip() for l in f.readlines()]

        import pickle
        with open('/home/hpaat/pcdet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb') as f:
            data = pickle.load(f)
        
        nusc = NuScenes(version='v1.0-trainval', dataroot='/home/hpaat/pcdet/data/nuscenes/v1.0-trainval/', verbose=True)
        nuscenes_path = '/home/hpaat/pcdet/data/nuscenes/v1.0-trainval/'
        
        with open("/home/hpaat/pcdet/data/nuscenes/v1.0-trainval/image_annotations.json") as f:
            image_annotations = json.load(f)
        
        for index, scene in enumerate(data):
            img = self.read_image(nuscenes_path + scene['cam_front_path'])
            save_image(img, '/home/hpaat/my_exp/MTrans/visualize/img_' + str(index) + '.jpg')

            H, W = img.shape[1:3]
            #point_cloud = self.read_point_cloud(nuscenes_path + scene['lidar_path'])    # N x 4
            scene_token = scene['token']
            sample_record = nusc.get('sample', scene_token)
            pointsensor_token = sample_record['data']['LIDAR_TOP']
            camera_token = sample_record['data']['CAM_FRONT']

            p2d_float, depth, _, mask, point_cloud = nusc.map_pointcloud_to_image(pointsensor_token, camera_token)
            p2d_float = p2d_float.T[:,:2]
            point_cloud = point_cloud.T
            point_cloud.astype(np.float32).tofile(path.join(out_3d_dir, f'{scene_token}.bin'))
            p2d_float.astype(np.float32).tofile(path.join(out_2d_dir, f'{scene_token}.bin'))

            rotys, dims, locs, img_size = scene['gt_boxes'][:,6], scene['gt_boxes'][:,3:6], \
                                          scene['gt_boxes'][:,0:3], [W,H]

            box_2d = self.get_box2d(nusc, scene, image_annotations) #self.encode_box2d(nusc, pointsensor_token, camera_token, rotys, dims, locs, img_size)
            with open('/home/hpaat/my_exp/MTrans/visualize/box2d_' + str(index) + '.npy', 'wb') as f:
                np.save(f, box_2d)

            ### build object-level dataset ###
            if self.cfg.split != 'test':
                object_labels, _ = self.prep_label(scene, box_2d)
            else:
                pass
                # if frame in test_all_labels.keys():
                #     object_labels = test_all_labels[frame]
                # else:
                #     continue

            overlap_matrix = self.build_overlap_matrix(object_labels)
            for index, obj in enumerate(object_labels):
                import pdb; pdb.set_trace() 
                obj['frame'] = scene['lidar_path'].split('/')[-1]
                # query sub cloud within the 2D box
                left, top, right, bottom = obj['box_2d'] # if len(obj['box_2d'].shape) == 1 else obj['box_2d'][index,:] 
                idx = np.logical_and.reduce([p2d_float[:, 0] > left, p2d_float[:, 1] > top, p2d_float[:, 0] < right, p2d_float[:, 1] < bottom])
                sub_cloud = point_cloud[idx]
                sub_cloud2d = p2d_float[idx]
                obj['sub_cloud'] = sub_cloud
                obj['sub_cloud2d'] = sub_cloud2d
                foreground_label = check_points_in_box(sub_cloud[:, :3], location=obj['location'], dimension=obj['dimensions'], yaw=obj['yaw'])
                obj['foreground_label'] = foreground_label

                overlap_boxes = [object_labels[j]['box_2d'] for j in overlap_matrix[index]]
                obj['overlap_boxes'] = overlap_boxes

                all_objects.append(obj)

#         for frame in tqdm(all_frames, desc=f"Processing {split} data"):
#             # preprocess frames, taking out the points within image scope, and their projected 2D coords
#             img = self.read_image(path.join(split_root, 'image_2', f'{frame}.png'))
#             H, W = img.shape[1:3]
#             point_cloud = self.read_point_cloud(path.join(split_root, 'velodyne', f'{frame}.bin'))
#             calib = self.read_calib(path.join(split_root, 'calib', f'{frame}.txt'))

# ############################################################################################################################################
#             # Modified/Added/Changed by Helbert PAAT
#             # Do the point cloud completion from the BtcDet
#             if self.pc_aug_btcdet:
#                 object_labels, btcdet_mask = self.read_label(path.join(split_root, 'label_2', f'n{frame}.txt'), calib)

#                 # If the frame is in the list of IDs, include the additional points
#                 id_addpoints_list_all = os.listdir(r"/home/hpaat/my_exp/BtcDet/data/kitti/detection3d/bm_50maxdist_2num_Car")
#                 start = str(int(frame)) + "_"
#                 id_addpoints_frame = sorted([filename for filename in id_addpoints_list_all if filename.startswith(start)])
#                 id_addpoints_frame = np.array(id_addpoints_frame)[btcdet_mask]
#                 if len(id_addpoints_frame) > 0:
#                     for index, file in enumerate(id_addpoints_frame):
#                         with open("/home/hpaat/my_exp/BtcDet/data/kitti/detection3d/bm_50maxdist_2num_Car/" + file, "rb") as f:
#                             point_cloud_addl = pickle.load(f)
#                         try: point_cloud_addl += object_labels[index]["location"]
#                         except: 
#                             print(frame, "\n")
#                             print(object_labels, "\n")
#                             print(id_addpoints_frame, "\n")
#                         point_cloud_addl = np.hstack((point_cloud_addl, np.zeros([point_cloud_addl.shape[0],1], point_cloud_addl.dtype)))
#                         point_cloud = np.vstack((point_cloud, point_cloud_addl))
# ############################################################################################################################################
            
#             p2d_float, depth = calib.velo_to_cam(point_cloud[:, :3])
#             x, y = p2d_float[:, 0], p2d_float[:, 1]
#             idx = np.logical_and.reduce([depth>=0, x>=0, x<W, y>=0, y<H])
#             point_cloud = point_cloud[idx]
#             p2d_float = p2d_float[idx]
#             point_cloud.astype(np.float32).tofile(path.join(out_3d_dir, f'{frame}.bin'))
#             p2d_float.astype(np.float32).tofile(path.join(out_2d_dir, f'{frame}.bin'))

#             ### build object-level dataset ###
#             if self.cfg.split!='test':
#                 object_labels, _ = self.read_label(path.join(split_root, 'label_2', f'{frame}.txt'), calib)
#             else:
#                 if frame in test_all_labels.keys():
#                     object_labels = test_all_labels[frame]
#                 else:
#                     continue
#             overlap_matrix = self.build_overlap_matrix(object_labels)
#             for i, obj in enumerate(object_labels):
#                 obj['frame'] = frame
#                 # query sub cloud within the 2D box
#                 left, top, right, bottom = obj['box_2d']
#                 idx = np.logical_and.reduce([p2d_float[:, 0]>left, p2d_float[:, 1]>top, p2d_float[:, 0]<right, p2d_float[:, 1]<bottom])
#                 sub_cloud = point_cloud[idx]
#                 sub_cloud2d = p2d_float[idx]
#                 obj['sub_cloud'] = sub_cloud
#                 obj['sub_cloud2d'] = sub_cloud2d
#                 # generate foreground label
#                 foreground_label = check_points_in_box(sub_cloud[:, :3], location=obj['location'], dimension=obj['dimensions'], yaw=obj['yaw'])
#                 obj['foreground_label'] = foreground_label

#                 overlap_boxes = [object_labels[j]['box_2d'] for j in overlap_matrix[i]]
#                 obj['overlap_boxes'] = overlap_boxes

#                 all_objects.append(obj)
        
        return all_objects

    def __getitem__(self, index):
        obj = copy.deepcopy(self.objects[index])
        obj = self.load_object_full_data(obj)
        return obj

    def load_object_full_data(self, obj):
        class_name = obj['class']
        obj['class_idx'] = self.cfg.classes.index(class_name) # TODO So this is always 0?
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