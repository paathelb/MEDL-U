import torch
import torchvision
from torch import nn
import torch.nn.functional as F
#from torch_scatter import scatter
from models.CNN_backbone.FPN import FPN
from models.modules.modality_mapper import img2pc
from utils.point_ops import build_image_location_map_single         # CHANGE
from models.modules.point_sa import AttentionPointEncoder
from loss import cal_diou_3d
import numpy as np
import math

from easydict import EasyDict
import scipy.stats as stats

# Evidential
from models.evidential import get_pred_evidential_aleatoric, get_pred_evidential_epistemic, UncertaintyHead, evidential_regression_loss, LinearNormalGamma, ShiftedSoftplus, get_pred_unc_one_parameter, \
    modified_mse

def wrap_to_pi(angles):
    wrapped_angles = torch.atan2(torch.sin(angles), torch.cos(angles))
    return wrapped_angles

def wrap_to_minus_pi_half_pi_half(angle):
    pi = torch.tensor(3.141592653589793238).cuda().float()
    wrapped_angle = torch.where(angle > pi/2, angle - pi, angle)
    wrapped_angle = torch.where(wrapped_angle <= -pi/2, angle + pi, wrapped_angle)
    return wrapped_angle

class MTrans(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.parameters_loaded = []             # record the names of parameters loaded from previous stage
        self.some_prints= cfgs.some_prints

        # Evidential DL parameters
        self.evi_uncertainty = cfgs.evi_uncertainty.setting
        self.evi_neurons = cfgs.evi_uncertainty.evi_neurons
        self.evi_loss_weight = cfgs.evi_uncertainty.loss_weight
        self.evi_lambda = cfgs.evi_uncertainty.evi_lambda
        self.evi_dim_only = cfgs.evi_uncertainty.evi_dim_only
        self.evi_loc_only = cfgs.evi_uncertainty.evi_loc_only
        self.evi_dimloc_only = cfgs.evi_uncertainty.evi_dimloc_only
        self.high_unc_reg = cfgs.evi_uncertainty.high_unc_reg
        self.shift_val = cfgs.evi_uncertainty.shift_val
        self.unc_act = cfgs.evi_uncertainty.unc_act
        self.comment = cfgs.evi_uncertainty.comment
        self.separate_heads = cfgs.evi_uncertainty.separate_heads
        self.yaw_loss = cfgs.evi_uncertainty.yaw_loss
        self.use_unprocessed_gt = cfgs.evi_uncertainty.use_unprocessed_gt
        self.choose_unc_idx = cfgs.evi_uncertainty.choose_unc_idx
        self.l_mse = cfgs.evi_uncertainty.l_mse
        self.unc_guided_iou_loss = cfgs.evi_uncertainty.unc_guided_iou_loss
        self.unc_guided_loss = cfgs.evi_uncertainty.unc_guided_loss
        self.rescale_unc = cfgs.evi_uncertainty.rescale_unc
        self.nll_weight = cfgs.evi_uncertainty.nll_weight

        # MTrans-Ensemble
        self.ensemble = cfgs.ensemble
        self.ensemble_lambda = cfgs.ensemble_lambda
        self.ensemble_dropout = cfgs.ensemble_dropout

        # MTrans- MCDO
        self.mcdo = cfgs.mcdo
        if self.mcdo:
            self.dropout_rate = cfgs.dropout_rate
            self.decay = cfgs.decay
        else:
            self.decay = None

        self.multi_evi = cfgs.multi_evi
        self.decouple_iou = cfgs.decouple_iou
        self.laplace_uncertainty = cfgs.laplace_uncertainty.setting
        self.lapl_lambda = cfgs.laplace_uncertainty.lambda_
        self.lapl_multi_unc = cfgs.laplace_uncertainty.multi_unc

        self.inc_lbox = cfgs.inc_lbox
        self.inc_lseg = cfgs.inc_lseg
        self.inc_ldepth = cfgs.inc_ldepth
        self.inc_lconf = cfgs.inc_lconf
        self.inc_ldir = cfgs.inc_ldir

        self.box_loss_weight = cfgs.box_loss_weight

        # CNN image encoder
        cimg, cpts = cfgs.POINT_ATTENTION.input_img_channel, cfgs.POINT_ATTENTION.input_pts_channel
        self.cnn = FPN(4, 128, cimg)

        # MAttn Transformer
        self.attention_layers = AttentionPointEncoder(cfgs.POINT_ATTENTION)
        self.xyzd_embedding = nn.Sequential(
            nn.Linear(3, cpts),
            nn.LayerNorm(cpts),
            nn.ReLU(inplace=True),
            nn.Linear(cpts, cpts)
        )
        self.unknown_f3d = nn.Parameter(torch.zeros(cpts))
        self.unknown_f3d = nn.init.normal_(self.unknown_f3d)
        
        hidden_size = cfgs.POINT_ATTENTION.hidden_size

        # Heads: 3D box regression
        self.foreground_head = nn.Sequential(
            nn.Linear(hidden_size+cimg+3, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.4),
            nn.Linear(512, 3)
        )
        
        if cfgs.POINT_ATTENTION.use_cls_token:
            if self.evi_uncertainty:
                if self.evi_dim_only or self.evi_loc_only:
                    self.box_head_1 = nn.Sequential(
                        nn.Linear(hidden_size+cimg+hidden_size, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=cfgs.box_drop),
                        nn.Linear(512, 7)
                    )                                                                   # prediction for the 7 parameters
                    self.box_head_evi_1 = nn.Sequential(
                        nn.Linear(hidden_size+cimg+hidden_size, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=cfgs.box_drop)
                    )       
                    self.box_head_evi_2 = UncertaintyHead(512, 'evidential_LNG_modified', 3)     # prediction for the uncertainties of the 3 parameters (either dim or loc)

                elif self.evi_dimloc_only:
                    if 'softplus_in_box_head_1' in self.comment:
                        self.box_head_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.Softplus(),
                            nn.Dropout(p=cfgs.box_drop),
                            nn.Linear(512, 7)
                        )   
                    else:
                        self.box_head_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            nn.Linear(512, 7)
                        )

                    if 'no_dropout_in_evi1' in self.comment and 'no_layernorm_in_evi1' in self.comment:        # prediction for the 7 parameters
                        self.box_head_evi_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.ReLU()
                        )
                    elif 'softplus_in_evi1' in self.comment and 'no_dropout_in_evi1' in self.comment and 'no_layernorm_in_evi1' in self.comment:         # prediction for the 7 parameters
                        self.box_head_evi_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.Softplus()
                        )
                    elif 'softplus_in_evi1' in self.comment and 'no_dropout_in_evi1' in self.comment:                                                  # prediction for the 7 parameters
                        self.box_head_evi_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.Softplus()
                        )       
                    elif 'softplus_in_evi1' in self.comment:
                        self.box_head_evi_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.Softplus(),
                            nn.Dropout(p=cfgs.box_drop)
                        ) 
                    else:                                                             # prediction for the 7 parameters
                        self.box_head_evi_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop)
                        )       

                    if self.separate_heads:
                        self.box_head_evi_2_loc0 = UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        self.box_head_evi_2_loc1 = UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        self.box_head_evi_2_loc2 = UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        self.box_head_evi_2_dim0 = UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        self.box_head_evi_2_dim1 = UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        self.box_head_evi_2_dim2 = UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                    else:    
                        self.box_head_evi_2 = UncertaintyHead(512, 'evidential_LNG_modified', 6, shift_val=self.shift_val, act=self.unc_act) 

                else:
                    self.box_head_1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            nn.Linear(512, 7)
                        )
                    if self.separate_heads:
                        self.box_head_evi_1_loc0 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )  
                        self.box_head_evi_1_loc1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )
                        self.box_head_evi_1_loc2 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )
                        self.box_head_evi_1_dim0 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )
                        self.box_head_evi_1_dim1 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )
                        self.box_head_evi_1_dim2 = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )
                        self.box_head_evi_1_rot = nn.Sequential(
                            nn.Linear(hidden_size+cimg+hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=cfgs.box_drop),
                            UncertaintyHead(512, 'evidential_LNG_modified', 1, shift_val=self.shift_val, act=self.unc_act) 
                        )
                    else:
                        self.box_head_evi_1 = nn.Sequential(
                                nn.Linear(hidden_size+cimg+hidden_size, self.evi_neurons),
                                nn.LayerNorm(self.evi_neurons),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=cfgs.box_drop)
                            )    
                        self.box_head_evi_2 = UncertaintyHead(self.evi_neurons, 'evidential_LNG_modified', 7)        # prediction for the 7 parameters each with 3 uncertainty components

            elif self.multi_evi:
                self.box_head_1 = nn.Sequential(
                    nn.Linear(hidden_size+cimg+hidden_size, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=cfgs.box_drop),
                    #nn.Linear(512, 7)
                )
                self.box_head_evi = UncertaintyHead(512, 'evidential', 7)
            elif self.laplace_uncertainty:
                if self.lapl_multi_unc:
                    self.lapl_unc_head = nn.Sequential(
                        nn.Linear(hidden_size+cimg+hidden_size, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=cfgs.box_drop),
                        nn.Linear(512, 14)
                    )
                else:
                    self.lapl_unc_head = nn.Sequential(
                        nn.Linear(hidden_size+cimg+hidden_size, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=cfgs.box_drop),
                        nn.Linear(512, 8)
                    )
            elif self.ensemble:
                self.ensemble_head = nn.Sequential(
                    nn.Linear(hidden_size+cimg+hidden_size, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=self.ensemble_dropout),
                    nn.Linear(512, 14)
                )     
                
                # Adding init decreases training performance. Why?
                # torch.nn.init.kaiming_normal_(self.ensemble_head[0].weight, mode='fan_in', nonlinearity='relu')
                # torch.nn.init.kaiming_normal_(self.ensemble_head[4].weight, mode='fan_in', nonlinearity='relu')
                # self.ensemble_head[0].bias.data.fill_(0)
                # self.ensemble_head[4].bias.data.fill_(0)
            elif self.mcdo:
                # Just the same
                pass
                # self.mcdo_head = nn.Sequential(
                #     nn.Linear(hidden_size+cimg+hidden_size, 512),
                #     nn.LayerNorm(512),
                #     nn.ReLU(inplace=True),
                #     nn.Dropout(p=self.dropout_rate),
                #     nn.Linear(512, 7),
                #     # nn.LayerNorm(256),
                #     # nn.ReLU(inplace=True),
                #     # nn.Dropout(p=self.dropout_rate),
                #     # nn.Linear(256, 7)
                # )
            else:
                pass

            self.box_head = nn.Sequential(
                nn.Linear(hidden_size+cimg+hidden_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=cfgs.box_drop),
                nn.Linear(512, 7)
            )
                    
            if self.decouple_iou:
                output = 4
            else:
                output = 3

            self.conf_dir_head = nn.Sequential(
                nn.Linear(hidden_size+cimg+hidden_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(512, output)
            )
        else:
            raise RuntimeError

        # Image transformations, apply data augmentation dynamically rather than in dataloader
        self.pred_transforms = torch.nn.Sequential(
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.train_transforms = torch.nn.Sequential(
            torchvision.transforms.RandomAutocontrast(p=0.5),
            torchvision.transforms.RandomAdjustSharpness(np.random.rand()*2, p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.3)], p=0.5),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def build_target_mask(self, B, N, starts, ends, device='cpu'):
        # starts: (B, ); ends: (B, )
        # fill each row with 'True', starting from the start_idx, to the end_idx
        mask = [([0]*starts[i] + [1] * (ends[i]-starts[i]) + [0] * (N-ends[i])) for i in range(B)]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        return mask                     # (B, N)

    def forward(self, data_dict):  
        image = data_dict['images']                                        # (B, C, H, W) # TODO Why only one image per item in the batch?
        overlap_masks = data_dict['overlap_masks']                         # (B, 1, H, W) # TODO Intuition?
        sub_cloud = data_dict['sub_clouds'][:, :, :3]                      # (B, N, 3) for x, y, z point cloud
        sub_cloud2d = data_dict['sub_clouds2d']                            # (B, N, 2) for projected point cloud
        ori_cloud2d = data_dict['ori_clouds2d']                            # (B, N, 2) for original 2D coords refer to the full image   # TODO Intuition?
        real_point_mask = data_dict['real_point_mask']                     # (B, N), 1-realpoint; 0-padding; 2-mask; 3-jitter           # TODO What is jitter?
        # TODO Is there an effect of the variations of mask with a lot of padding 0 or with no padding at all?
        foreground_label = data_dict['foreground_label']                   # (B, N), 1-foreground; 0-background; 2-unknown              # TODO Why padding points are "unknown"?
        device = sub_cloud.device
        pred_dict = {'batch_size': torch.Tensor([image.size(0)]).cuda()}
        
        impact_points_mask = real_point_mask == 1                           # (B, N), points with known 3D coords, unmasked, unjittered     # no padding
        unmaksed_known_points = (impact_points_mask) + (real_point_mask==3) # (B, N), points has 3D coords, no masked, no padding
        nonpadding_points = (unmaksed_known_points) + (real_point_mask==2)  # (B, N), points known, no padding

        # Normalize point cloud # Getting the average of real point clouds per batch
        sub_cloud_center = (sub_cloud * impact_points_mask.unsqueeze(-1)).sum(dim=1) / (impact_points_mask.sum(dim=1, keepdim=True) + 1e-6)     # (B, 3)
        
        # Only norm x&y coords
        # sub_cloud_center[:,-1] = 0                                         # B x 3 # TODO Intuition?
        data_dict['locations'] = data_dict['locations'] - sub_cloud_center       # B x 3 # TODO What is locations? the center
        sub_cloud = sub_cloud - sub_cloud_center.unsqueeze(1)              # B x N x 3  # TODO Intuition?
        data_dict['sub_clouds'][:, :, :3] = data_dict['sub_clouds'][:, :, :3] - sub_cloud_center.unsqueeze(1) # B x N x 4
        sub_cloud = sub_cloud * (nonpadding_points).unsqueeze(-1)          # B x N x 3
        pred_dict['subcloud_center'] = sub_cloud_center                    # B x 3
        
        # Randomly augment point cloud with 50% probability
        if self.training:
            random_shift = (torch.rand(sub_cloud.size(0), 1, 3, device=sub_cloud.device) * torch.tensor([[[2, 2, 0.5]]], device=sub_cloud.device) \
                            - torch.tensor([[[1, 1, 0.25]]], device=sub_cloud.device))                                      # B x 1 x 3 # TODO Intuition?
            random_shift = random_shift * torch.randint(0, 2, (sub_cloud.size(0), 1, 1), device=random_shift.device).bool() # B x 1 x 3 # 50% gets augmented 
            sub_cloud = sub_cloud + random_shift                                # B x N x 3  # TODO What is the relevance of this? Note that the shift for x,y has max of 1 and for z max of 0.25
            data_dict['locations'] = data_dict['locations'] + random_shift.squeeze(1) # B x 3

        # Randomly scale point cloud
        if self.training:
            random_scale = (torch.rand(sub_cloud.size(0), 1, 3, device=sub_cloud.device) * 0.2 + 0.9)      # B x 1 x 3 # Max 1.10 Min 0.90
            random_idx = torch.randint(0, 2, (sub_cloud.size(0), 1, 1), device=random_scale.device).bool() # B x 1 x 1
            random_scale = random_scale * random_idx + 1 * (~random_idx)                                   # B x 1 x 3

            sub_cloud = sub_cloud * random_scale                                    # B x N x 3
            data_dict['locations'] = data_dict['locations'] * random_scale.squeeze(1)     # B x 3
            data_dict['dimensions'] = data_dict['dimensions'] * random_scale.squeeze(1)   # B x 3 # TODO Check how data_dict.dimensions is calculated

        # Random flip
        if self.training:
            # flip y coords
            random_idx = torch.randint(0, 2, (sub_cloud.size(0), 1), device=sub_cloud.device) * 2 - 1  # B x 1  # Either 1 or -1
            sub_cloud[:, :, 1] = sub_cloud[:, :, 1] * random_idx                                       # B x N x 3 # TODO Intuition?
            data_dict['locations'][:, 1] = data_dict['locations'][:, 1] * random_idx.squeeze(1)              # B x 3
            data_dict['yaws'] = data_dict['yaws'] * random_idx                                               # B x 1 # TODO Intuition?
            data_dict['sub_clouds'] = sub_cloud                                                           # B x N x 3

            flipped_image = torch.flip(image, dims=(-1,))      # TODO Flip upwards or rightward?                          # B x C x H X W
            image = image * (random_idx != -1).view(-1, 1, 1, 1) + flipped_image * (random_idx == -1).view(-1, 1, 1, 1)   # B x C x H X W
            # The way we  flip the y coords is the way we also flip the image
            flipped_x = image.size(-1) - 1 - sub_cloud2d[:, :, 0]                                       # B x N  # TODO Intuition?
            sub_cloud2d[:,:,0] = sub_cloud2d[:,:,0] * (random_idx != -1) + flipped_x * (random_idx==-1)   # B x N
            # The way we  flip the x coords is the way we also flip the image
            data_dict['images'] = image
            data_dict['sub_cloud2d'] = sub_cloud2d

        pred_dict['gt_coords_3d'] = sub_cloud

        # Randomly augment input image
        if self.training:
            image = torch.cat([self.train_transforms(image), overlap_masks], dim=1) # B x 4 x H x W
        else:
            # Normalize image
            image = torch.cat([self.pred_transforms(image), overlap_masks], dim=1)

        # -------------------------------------------------------- 1. Extract information of images --------------------------------------------------------
        B, _, H, W = image.size()
        image_features, _ = self.cnn(image)         # Deep Neural Network of Conv2d, BatchNorm2d, and ReLU      # B X 512 X H X W
        
        # -------------------------------------------------------- 2. Build new cloud, which contains blank slots to be interpolated -----------------------
        B, N, _ = sub_cloud.size()
        scale = self.cfgs.sparse_query_rate
        qH, qW = H//scale, W//scale                 # TODO Intuition?

        # Hide and jitter point cloud
        jittered_cloud = sub_cloud.clone()                                              # B x N x 3
        jittered_cloud[(real_point_mask==3).unsqueeze(-1).repeat(1, 1, 3)] += torch.rand((real_point_mask==3).sum() * 3, device=sub_cloud.device) * 0.1 - 0.05      # Max 0.05 / Min -0.05    # TODO What if (real_point_mask==3).sum() == 0?
        jittered_cloud = jittered_cloud * (unmaksed_known_points.unsqueeze(-1))         # B x N x 3
                
        key_c2d = sub_cloud2d                                                                                   # (B, N, 2)
        key_f2d = img2pc(image_features, key_c2d).transpose(-1, -2)                                             # (B, N, Ci)
        key_f3d = self.xyzd_embedding(jittered_cloud) * (unmaksed_known_points.unsqueeze(-1)) + \
                  self.unknown_f3d.view(1, 1, -1).repeat(B, N, 1) * (~unmaksed_known_points.unsqueeze(-1))      # B x N x Ci
        query_c2d = (build_image_location_map_single(qH, qW, device)*scale).view(1, -1, 2).repeat(B, 1, 1)      # (B, qH*qW, 2)         # Changed # Uniform sampling from the image
        query_f2d = img2pc(image_features, query_c2d).transpose(-1, -2)                                         # (B, qH*qW, Ci)
        query_f3d = self.unknown_f3d.view(1, 1, -1).repeat(B, query_f2d.size(1), 1)                             # (B, qH*qW, Ci)        # NOTE Consider downsampling the points (even for those with more than 512 points) and get key_c2d, key_f2d, and key_f3d.

        # -------------------------------------------------------- 3. Self-attention to decode missing 3D features ----------------------------------------
        # Only unmasked known foreground will be attended
        attn_mask = unmaksed_known_points                                                                                               # B x N
        query_f3d, key_f3d, cls_f3d = self.attention_layers(query_c2d, query_f2d, query_f3d, key_c2d, key_f2d, key_f3d, attn_mask)      # B x (M or N or 1) x hidden_size    #, data=data_dict if self.cfgs.visualize_attn else None)
        
        # Same xyz head and foreground head are used for pred and query
        # Pred foreground logits
        pred_key_coords_3d = self.xyz_head(key_f3d)                                                # (B, N, 3)
        pred_dict['pred_coords_3d'] = pred_key_coords_3d                                           # B x N x 3   # Predicted 3D points?
        diff_xyz = (pred_key_coords_3d.detach() - sub_cloud) * nonpadding_points.unsqueeze(-1)     # B x N x 3   # Why nonpadding?

        pred_key_foreground = self.foreground_head(torch.cat([key_f2d, key_f3d, diff_xyz], dim=-1))   # (B, N, 2)
        pred_dict['pred_foreground_logits'] = pred_key_foreground                                     # (B, N, 2)

        # Query foreground logits
        pred_query_coords_3d = self.xyz_head(query_f3d)                                               # B x M x 3
        pred_dict['enriched_points'] = pred_query_coords_3d                                           # B x M x 3

        pred_query_foreground = self.foreground_head(torch.cat([query_f2d, query_f3d, torch.zeros_like(pred_query_coords_3d)], dim=-1))     # B x M x 2
        pred_dict['enriched_foreground_logits'] = pred_query_foreground
        
        # -------------------------------------------------------- 4. Predict 3D box ----------------------------------------------------------------------
        # Norm center
        all_points = torch.cat([sub_cloud], dim=1)                                                                                          # B x N x 3
        all_forground_mask = torch.cat([pred_key_foreground], dim=1).argmax(dim=-1, keepdim=True) * unmaksed_known_points.unsqueeze(-1)     # B x N x 1
        seg_center = (all_points * all_forground_mask).sum(dim=1) / ((all_forground_mask).sum(dim=1) + 1e-6)                                # B x 3 # Get the average per batch item

        data_dict['locations'] = data_dict['locations'] - seg_center        # B x 3     # TODO Why update this? Intuition?
        pred_dict['second_offset'] = seg_center                             # B x 3

        # Preparing inputs for the box head for box prediction
        # Make the logits more discriminative
        query_fore_logits = (pred_query_foreground * 5).softmax(dim=-1)           # B x M x 2
        key_fore_logits = (pred_key_foreground * 5).softmax(dim=-1)               # B x N x 2
        global_f3d_num = (query_f3d * query_fore_logits[:, :, 1:2]).sum(dim=1) + (key_f3d * key_fore_logits[:, :, 1:2]).sum(dim=1)      # Num: B x hidden_size
        global_f3d_denom = query_fore_logits[:, :, 1:2].sum(dim=1) + key_fore_logits[:, :, 1:2].sum(dim=1)                              # Denom: B x 1
        global_f3d = global_f3d_num / (global_f3d_denom + 1e-10)                                        # B x hidden_size 
        global_f2d = torch.nn.AdaptiveAvgPool2d((1,1))(image_features).squeeze(-1).squeeze(-1)          # B x N
        
        box_feature = torch.cat([cls_f3d.squeeze(1), global_f3d, global_f2d], dim=-1)                   # B x 2048

        if self.evi_uncertainty:
            if self.evi_dim_only or self.evi_loc_only or self.evi_dimloc_only:
                box = self.box_head_1(box_feature)                                                # B x 7
                box_evi_features = self.box_head_evi_1(box_feature)                                         # Tuple of length 4: each is a tensor of shape 4 x 7

                if self.separate_heads:
                    box_uncertainty_loc0 = self.box_head_evi_2_loc0(box_evi_features)
                    box_uncertainty_loc1 = self.box_head_evi_2_loc1(box_evi_features)
                    box_uncertainty_loc2 = self.box_head_evi_2_loc2(box_evi_features)
                    box_uncertainty_dim0 = self.box_head_evi_2_dim0(box_evi_features)
                    box_uncertainty_dim1 = self.box_head_evi_2_dim1(box_evi_features)
                    box_uncertainty_dim2 = self.box_head_evi_2_dim2(box_evi_features)
                    pred_dict['box_uncertainty'] = (box_uncertainty_loc0, box_uncertainty_loc1, box_uncertainty_loc2, box_uncertainty_dim0, box_uncertainty_dim1, box_uncertainty_dim2)
                else:
                    box_uncertainty = self.box_head_evi_2(box_evi_features)
                    pred_dict['box_uncertainty'] = box_uncertainty                                              # B x 3 x S
            else:
                box = self.box_head_1(box_feature)  
                if self.separate_heads:
                    box_uncertainty_loc0 = self.box_head_evi_1_loc0(box_feature)
                    box_uncertainty_loc1 = self.box_head_evi_1_loc1(box_feature)
                    box_uncertainty_loc2 = self.box_head_evi_1_loc2(box_feature)
                    box_uncertainty_dim0 = self.box_head_evi_1_dim0(box_feature)
                    box_uncertainty_dim1 = self.box_head_evi_1_dim1(box_feature)
                    box_uncertainty_dim2 = self.box_head_evi_1_dim2(box_feature)
                    box_uncertainty_rot = self.box_head_evi_1_rot(box_feature)
                    pred_dict['box_uncertainty'] = (box_uncertainty_loc0, box_uncertainty_loc1, box_uncertainty_loc2, box_uncertainty_dim0, box_uncertainty_dim1, box_uncertainty_dim2, box_uncertainty_rot)
                else:                                              # B x 7
                    box_evi_features = self.box_head_evi_1(box_feature)                                                    # Tuple of length 4: each is a tensor of shape 4 x 7
                    box_uncertainty = self.box_head_evi_2(box_evi_features)
                    pred_dict['box_uncertainty'] = box_uncertainty                                              # B x 3 x 7

            # conf_dir_pred_feature = self.conf_dir_head_1(box_feature)                                 # B x 3
            # conf_dir_preds = self.conf_dir_head_evi(conf_dir_pred_feature)
            # conf_dir_preds_uncertainty = conf_dir_preds[1:]
            # conf_dir_pred = conf_dir_preds[0]
            # pred_dict['conf_dir_preds_uncertainty'] = conf_dir_preds_uncertainty

            location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]                            # B x 3 # B x 3 # B x 1

            if self.use_unprocessed_gt:
                pred_dict['location_evi'] = location         # B x 3
                pred_dict['dimension_evi'] = dimension        # B x 3
                yaw_evi = torch.tanh(yaw)                                                  # B x 1 # TODO Intuition? 
                yaw_evi = torch.arcsin(yaw_evi)                                           # B x 1 # Why arcsin(tanh(yaw))
                pred_dict['yaw_evi'] = yaw_evi              # B x 1
        elif self.laplace_uncertainty:
            if self.lapl_multi_unc:
                box = self.lapl_unc_head(box_feature)  
                location, dimension, yaw, lapl_unc = box[:, 0:3], box[:, 3:6], box[:, 6:7], box[:, 7:]                      # B x 3 # B x 3 # B x 1  
            else:
                box = self.lapl_unc_head(box_feature)  
                location, dimension, yaw, lapl_unc = box[:, 0:3], box[:, 3:6], box[:, 6:7], box[:, 7:8]                      # B x 3 # B x 3 # B x 1  
            pred_dict['lapl_unc'] = lapl_unc
        elif self.ensemble:
            box = self.ensemble_head(box_feature)  
            location, dimension, yaw, sigmas = box[:, 0:3], box[:, 3:6], box[:, 6:7], box[:, 7:]                            # B x 3 # B x 3 # B x 1 # B x 7
            pred_dict['sigmas'] = sigmas
        elif self.mcdo:
            # box = self.mcdo_head(box_feature)  
            box = self.box_head(box_feature)        
            location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]   
        else:
            box = self.box_head(box_feature)     
            location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]                                                # B x 3 # B x 3 # B x 1  
            
        conf_dir_pred = self.conf_dir_head(box_feature)        
        
        # Decouple IoU
        if self.decouple_iou:
            direction, purity, integrity = conf_dir_pred[:, 0:2], conf_dir_pred[:, 2], conf_dir_pred[:, 3]      # B x 2 # B x 1   
            purity = torch.nn.Sigmoid()(purity) 
            integrity = torch.nn.Sigmoid()(integrity) 
            confidence = 1 / (1/purity + 1/integrity - 1)
        else:
            direction, confidence = conf_dir_pred[:, 0:2], conf_dir_pred[:, 2:3] 
            confidence = torch.nn.Sigmoid()(confidence)                             # B x 1             # Can this be used for the uncertainty estimation?




        ########################################################################################################################################################################
        
        # Post-processing every forward pass
        dim_anchor = torch.tensor(self.cfgs.anchor, device=device).view(1, 3)           # 1 x 3 # [4.0000, 1.6000, 1.5000]
        da = torch.norm(dim_anchor[:, :2], dim=-1, keepdim=True)                        # 1 x 1 # da = sqrt(4.0**2 + 1.6**2) = 4.3081
        ha = dim_anchor[:, 2:3]                                                         # 1 x 1
        pred_loc = location * torch.cat([da, da, ha], dim=-1)                           # B x 3 # TODO What is the intuition for this?
        pred_dim = torch.exp(dimension) * dim_anchor                                    # B x 3 # TODO Why exp? Intuition?

        if 'pi' in self.yaw_loss[-1]:
            try: pred_yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            except:
                if 'min_val' in self.yaw_loss[0]:
                    pred_yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)+1e-6)
                elif 'cons' in self.yaw_loss[0]:
                    pass
        else:
            pred_yaw = torch.tanh(yaw)                                                  # B x 1 # TODO Intuition? 
            pred_yaw = torch.arcsin(pred_yaw)                                           # B x 1 # Why arcsin(tanh(yaw))
            

        pred_dict['location'] = pred_loc         # B x 3
        pred_dict['dimension'] = pred_dim        # B x 3
        pred_dict['yaw'] = pred_yaw              # B x 1
        pred_dict['direction'] = direction       # B x 2
        pred_dict['conf'] = confidence           # B x 1            # B x 2 if decoupled IoU

        if self.decouple_iou:
            pred_dict['purity'] = purity
            pred_dict['integrity'] = integrity  
        
        return pred_dict                         # Keys: 'batch_size', 'subcloud_center', 'gt_coords_3d', 'pred_coords_3d', 'pred_foreground_logits', 'enriched_points', 
                                                 # 'enriched_foreground_logits', 'second_offset', 'location', 'dimension', 'yaw', 'direction', 'conf'
     
    def get_weighted_loss(self, loss_val, weights):
        weighted_loss = 0
        for idx, weight in enumerate(weights):
            weighted_loss += weight * loss_val[idx]
        return weighted_loss

    def get_loss(self, pred_dict, data_dict, rank, weights=None):

        if weights is not None:
            if all(elem==1 for elem in weights):
                weights = None

        loss_dict = {}
        B = pred_dict['batch_size'].sum().item()
        has_label = data_dict['use_3d_label']                      # (B)
        real_point_mask = data_dict['real_point_mask']             # (B, N)


        # ----------------------------------------------------------------------------- 1. Foreground loss ----------------------------------------------------------------------------- 
        segment_logits = pred_dict['pred_foreground_logits'].transpose(-1,-2)       # (B, 2, N)     # For foreground and background
        gt_segment_label = data_dict['foreground_label']                            # (B, N)        # How did we get data_dict.foreground_label?
        
        # Loss only for those have 3D label
        segment_gt, segment_logits = gt_segment_label[has_label], segment_logits[has_label]    # Bl x N   # Bl x 2 x N
        criterion_segment = nn.CrossEntropyLoss(reduction='none', ignore_index=2)              # Ignore label of 2
        loss_segment = criterion_segment(segment_logits, segment_gt)                           # Bl x N    # TODO Why segment_logits does not sum up to 1?
        
        # Balance foreground and background, take mean across batch samples
        lseg = 0
        if (segment_gt==1).sum() > 0:
            lseg = lseg + (loss_segment * (segment_gt==1)).sum(dim=-1) / ((segment_gt==1).sum(dim=-1) + 1e-6) # Bl # Average segment_loss for foreground
        if (segment_gt==0).sum() > 0:
            lseg = lseg + (loss_segment * (segment_gt==0)).sum(dim=-1) / ((segment_gt==0).sum(dim=-1) + 1e-6) # Bl # Average segment_loss for background
        
        if self.unc_guided_loss:
            l_seg_unc = lseg.clone()

        if weights is None:
            loss_segment = lseg.mean()          
        else:
            loss_segment = self.get_weighted_loss(lseg, weights)

        # Add Dice Loss to Loss Segment     
        segment_prob = segment_logits.softmax(dim=1)[:, 1, :]                                           # Bl x N        # NOTE Probability of being classified as foreground
        inter = 2 * (segment_prob * (segment_gt==1)).sum(dim=-1) + 1e-6                                 # Bl            #  Intuition? 
        uni = (segment_prob * (segment_gt != 2)).sum(dim=-1) + (segment_gt == 1).sum(dim=-1) + 1e-6     # Bl            #  Intuition? 
        dice_loss = 1 - inter/uni                                                   # Bl                                #  Intuition? 

        if self.unc_guided_loss:
            l_dice_unc = dice_loss.clone()

        if weights is None:
            dice_loss = dice_loss.mean()          
        else:
            dice_loss = self.get_weighted_loss(dice_loss, weights)

        loss_segment = loss_segment + dice_loss                         

        # Metric: Segment IoU
        segment_pred = segment_logits.argmax(dim=1) * (segment_gt != 2)             # Bl x N 
        intersection = (segment_pred * (segment_gt == 1)).sum(dim=1)                # Bl
        union = ((segment_pred + (segment_gt == 1)).bool()).sum(dim=1) + 1e-10      # Bl

        seg_iou = (intersection / union).mean()                                     # Single value
        loss_dict['loss_segment'] = (loss_segment.item(), B, 'losses')
        loss_dict['segment_iou'] = (seg_iou.item(), B, 'segment_iou')
        




        #  ----------------------------------------------------------------------------- 2. Depth loss  ----------------------------------------------------------------------------- 
        gt_coords = pred_dict['gt_coords_3d']                                   # (B, N, 3) # sub_cloud
        pred_coords = pred_dict['pred_coords_3d']                               # (B, N, 3) # key_f3d passed into a downsampling head
        loss_mask = pred_dict['pred_foreground_logits'].argmax(dim=-1).float()  # B x N # ignore background # Based on key
        loss_mask = (loss_mask) * (real_point_mask != 0)                        # B x N # ignore padding # TODO Why consider all items in batch?  # Why ignore padding?

        criterion_depth = nn.SmoothL1Loss(reduction='none')
        loss_depth = criterion_depth(pred_coords, gt_coords).sum(dim=-1)        # B x N # TODO Intuition?
        loss_depth = loss_depth * loss_mask                                     # B x N

        # Balance mask/jitter/impact points
        l = 0
        l = l + (loss_depth * (real_point_mask == 1)).sum(dim=1) / (((real_point_mask == 1)*loss_mask).sum(dim=1) + 1e-6) * 0.1     # B # Real points # TODO Why multiply by 0.10?
        l = l + (loss_depth * (real_point_mask == 2)).sum(dim=1) / (((real_point_mask == 2)*loss_mask).sum(dim=1) + 1e-6)           # B # Mask points # No *0.1 # Bigger contribution to loss compared to real points
        assert (real_point_mask != 3).all()                                     # TODO Why? What is the intuition? So we don't expect to have label 3 = jitter at all?
        
        if self.unc_guided_loss:
            l_depth_unc = l.clone()

        if weights is None:
            loss_depth = l.mean()          
        else:
            loss_depth = self.get_weighted_loss(l, weights)
        
        # Metric: mean xyz err
        err_dist = torch.norm(pred_coords - gt_coords, dim=-1)                      # B x N
        err_dist = ((err_dist * (loss_mask == 1) * (real_point_mask == 2)).sum(dim=-1) / (((loss_mask == 1) * (real_point_mask == 2)).sum(dim=-1) + 1e-6))     # B
        loss_dict['loss_depth'] = (loss_depth.item(), B, 'losses')
        loss_dict['err_dist'] = (err_dist.mean().item(), B, 'err_dist')             # real_point_mask==2
        




        #  ----------------------------------------------------------------------------- 3. Box loss ----------------------------------------------------------------------------- 
        pred_loc, pred_dim, pred_yaw = pred_dict['location'], pred_dict['dimension'], pred_dict['yaw']  # B x 3 # B x 3 # B x 1
        gt_loc, gt_dim, gt_yaw = data_dict['locations'], data_dict['dimensions'], data_dict['yaws']     # B x 3 # B x 3 # B x 1
        pred_loc, pred_dim, pred_yaw = pred_loc[has_label], pred_dim[has_label], pred_yaw[has_label]    # Bl x 3 # Bl x 3 # Bl x 1
        gt_loc, gt_dim, gt_yaw = gt_loc[has_label], gt_dim[has_label], gt_yaw[has_label]                # Bl x 3 # Bl x 3 # Bl x 1
        gt_boxes = torch.cat([gt_loc, gt_dim, gt_yaw], dim=-1)                                          # Bl x 7
        pred_boxes = torch.cat([pred_loc, pred_dim, pred_yaw], dim=-1)                                  # Bl x 7
        num_gt_samples = has_label.sum()
        diff_yaw = torch.sin(pred_yaw - gt_yaw)                                                         # Bl x 1    # TODO Intuition as to why we need to get the sin of it
        
        if self.decouple_iou:
            l_iou, iou3d, purity, integrity, iou2d = cal_diou_3d(pred_boxes.unsqueeze(1), gt_boxes.unsqueeze(1), decouple_iou=self.decouple_iou)               # Bl x 1    # Bl x 1    # Bl x 1          
        else:
            l_iou, iou3d, iou2d = cal_diou_3d(pred_boxes.unsqueeze(1), gt_boxes.unsqueeze(1))

        if self.unc_guided_loss or self.unc_guided_iou_loss:
            l_iou_unc = l_iou.clone()

        if self.laplace_uncertainty:
            if self.lapl_multi_unc:
                gt_boxes[:,6] = wrap_to_minus_pi_half_pi_half(gt_boxes[:,6])
                lv_loc0, lv_loc1, lv_loc2, lv_dim0, lv_dim1, lv_dim2, lv_yaw = pred_dict['lapl_unc'][:,0].clamp(min=-10, max=0), \
                                                                               pred_dict['lapl_unc'][:,1].clamp(min=-10, max=0), \
                                                                               pred_dict['lapl_unc'][:,2].clamp(min=-10, max=0), \
                                                                               pred_dict['lapl_unc'][:,3].clamp(min=-10, max=0), \
                                                                               pred_dict['lapl_unc'][:,4].clamp(min=-10, max=0), \
                                                                               pred_dict['lapl_unc'][:,5].clamp(min=-10, max=0), \
                                                                               pred_dict['lapl_unc'][:,6].clamp(min=-10, max=0)
                criterion_lapl = nn.SmoothL1Loss(reduction='none')
                l_loc0 =  (1.4142 * torch.exp(-lv_loc0) * criterion_lapl(gt_boxes[:,0], pred_boxes[:,0]) + self.lapl_lambda * lv_loc0).mean()
                l_loc1 =  (1.4142 * torch.exp(-lv_loc1) * criterion_lapl(gt_boxes[:,1], pred_boxes[:,1]) + self.lapl_lambda * lv_loc1).mean()
                l_loc2 =  (1.4142 * torch.exp(-lv_loc2) * criterion_lapl(gt_boxes[:,2], pred_boxes[:,2]) + self.lapl_lambda * lv_loc2).mean()
                l_dim0 =  (1.4142 * torch.exp(-lv_dim0) * criterion_lapl(gt_boxes[:,3], pred_boxes[:,3]) + self.lapl_lambda * lv_dim0).mean()
                l_dim1 =  (1.4142 * torch.exp(-lv_dim1) * criterion_lapl(gt_boxes[:,4], pred_boxes[:,4]) + self.lapl_lambda * lv_dim1).mean()
                l_dim2 =  (1.4142 * torch.exp(-lv_dim2) * criterion_lapl(gt_boxes[:,5], pred_boxes[:,5]) + self.lapl_lambda * lv_dim2).mean()
                l_yaw =  (1.4142 * torch.exp(-lv_yaw) * criterion_lapl(gt_boxes[:,6], pred_boxes[:,6]) + self.lapl_lambda * lv_yaw).mean()

                loss_dict['lapl_unc_x'] = ((torch.exp(lv_loc0)).abs().mean().item(), num_gt_samples, 'iou')
                loss_dict['lapl_unc_y'] = ((torch.exp(lv_loc1)).abs().mean().item(), num_gt_samples, 'iou')
                loss_dict['lapl_unc_z'] = ((torch.exp(lv_loc2)).abs().mean().item(), num_gt_samples, 'iou')
                loss_dict['lapl_unc_l'] = ((torch.exp(lv_dim0)).abs().mean().item(), num_gt_samples, 'iou')
                loss_dict['lapl_unc_w'] = ((torch.exp(lv_dim1)).abs().mean().item(), num_gt_samples, 'iou')
                loss_dict['lapl_unc_h'] = ((torch.exp(lv_dim2)).abs().mean().item(), num_gt_samples, 'iou')
                loss_dict['lapl_unc_rot'] = ((torch.exp(lv_yaw)).abs().mean().item(), num_gt_samples, 'iou')
            else:
                # Only single uncertainty for each object
                log_variance = pred_dict['lapl_unc'].clamp(min=-10, max=0).reshape(-1)
                l_iou =  1.4142 * torch.exp(-log_variance) * l_iou + self.lapl_lambda * log_variance 
                loss_dict['lapl_unc_checker'] = ((torch.exp(log_variance)).abs().mean().item(), num_gt_samples, 'iou')
        elif self.ensemble:
            gt_boxes[:,6] = wrap_to_minus_pi_half_pi_half(gt_boxes[:,6])

            log_variance = pred_dict['sigmas']                                            #.clamp(min=-10, max=0)
            criterion_lapl = nn.SmoothL1Loss(reduction='none')

            l_loc0 =  ((log_variance[:,0] + criterion_lapl(gt_loc[:,0], pred_loc[:,0])/torch.exp(log_variance[:,0])) / 2.0).mean()
            l_loc1 =  ((log_variance[:,1] + criterion_lapl(gt_loc[:,1], pred_loc[:,1])/torch.exp(log_variance[:,1])) / 2.0).mean()
            l_loc2 =  ((log_variance[:,2] + criterion_lapl(gt_loc[:,2], pred_loc[:,2])/torch.exp(log_variance[:,2])) / 2.0).mean()
            l_dim0 =  ((log_variance[:,3] + criterion_lapl(gt_dim[:,0], pred_dim[:,0])/torch.exp(log_variance[:,3])) / 2.0).mean()
            l_dim1 =  ((log_variance[:,4] + criterion_lapl(gt_dim[:,1], pred_dim[:,1])/torch.exp(log_variance[:,4])) / 2.0).mean()
            l_dim2 =  ((log_variance[:,5] + criterion_lapl(gt_dim[:,2], pred_dim[:,2])/torch.exp(log_variance[:,5])) / 2.0).mean()
            l_yaw =   ((log_variance[:,6] + criterion_lapl(gt_yaw, pred_yaw)/torch.exp(log_variance[:,6])) / 2.0).mean()
            
            var = torch.exp(log_variance)           # Useful for inference
            loss_dict['ensemble_var_checker'] = ((torch.exp(log_variance)).abs().mean().item(), num_gt_samples, 'iou')
 
        if weights is None:
            loss_box = l_iou.mean()          
        else:
            loss_box = self.get_weighted_loss(l_iou, weights)

        # Loss for direction/ yaw angle                     # NOTE It did not use the predicted yaw
        gt_dir = self.clamp_orientation_range(gt_yaw)                                                   # Bl x 1 # TODO What is the intuition in doing this? 
        gt_dir = ((gt_dir >= -np.pi/2) * (gt_dir < np.pi/2)).long().squeeze(-1)                         # Bl
        pred_dir = pred_dict['direction'][has_label]  
        criterion_dir = torch.nn.CrossEntropyLoss(reduction='none')                                     # Bl x 2
        loss_dir = criterion_dir(pred_dir, gt_dir)          

        if self.unc_guided_loss:
            l_dir_unc = loss_dir.clone()

        if weights is None:
            loss_dir = loss_dir.mean()   
        else:
            loss_dir = self.get_weighted_loss(loss_dir, weights)

        acc_dir = (pred_dir.argmax(dim=-1) == gt_dir).sum() / num_gt_samples

        # Loss for confidence
        if self.decouple_iou:
            pred_purity = pred_dict['purity'][has_label] 
            pred_integrity = pred_dict['integrity'][has_label]  
            confidence = pred_dict['conf'][has_label]  
            criterion_conf = torch.nn.SmoothL1Loss(reduction='none')
            loss_purity = criterion_conf(pred_integrity, integrity.squeeze()).mean()            #-(purity.squeeze() * torch.log(pred_purity) + (1-purity).squeeze() * torch.log(1-pred_purity)).mean()
            loss_integrity = criterion_conf(pred_purity, purity.squeeze()).mean()               #-(integrity.squeeze() * torch.log(pred_integrity) + (1-integrity).squeeze() * torch.log(1-pred_integrity)).mean()
            loss_conf = criterion_conf(confidence, iou3d.squeeze())                             #-(iou3d.squeeze() * torch.log(confidence) + (1-iou3d) * torch.log(1-confidence))
        else:   
            confidence = pred_dict['conf'][has_label]  
            criterion_conf = torch.nn.SmoothL1Loss(reduction='none')                                        # Bl x 1
            loss_conf = criterion_conf(confidence, iou3d)   
        
        if self.unc_guided_loss:
            l_conf_unc = loss_conf.clone()                                                              # Single value

        if weights is None:
            loss_conf = loss_conf.mean()   
        else:
            loss_conf = self.get_weighted_loss(loss_conf, weights)
                                                               
        err_conf = ((confidence - iou3d).abs().sum() / num_gt_samples)                                  # Single value # TODO Why define it this way?
        
        try: assert not iou3d.isnan().any()     # Why pred_boxes all nan?
        except: import pdb; pdb.set_trace() 
        

        #  --------------------------------------------------------------- 4. Evidential Regression Loss -------------------------------------------------------------------------
        if self.evi_uncertainty:
            if self.evi_dim_only:
                evidential_loss = evidential_regression_loss(gt_boxes[:,3:6], (pred_dim,) + pred_dict['box_uncertainty'], self.evi_lambda)
            elif self.evi_loc_only:
                evidential_loss = evidential_regression_loss(gt_boxes[:,:3], (pred_loc,) + pred_dict['box_uncertainty'], self.evi_lambda)
            elif self.evi_dimloc_only:
                if self.separate_heads:
                    evidential_loss_loc0 = evidential_regression_loss(gt_boxes[:,0], (pred_boxes[:,0],) + pred_dict['box_uncertainty'][0], self.evi_lambda)
                    evidential_loss_loc1 = evidential_regression_loss(gt_boxes[:,1], (pred_boxes[:,1],) + pred_dict['box_uncertainty'][1], self.evi_lambda)
                    evidential_loss_loc2 = evidential_regression_loss(gt_boxes[:,2], (pred_boxes[:,2],) + pred_dict['box_uncertainty'][2], self.evi_lambda)
                    evidential_loss_dim0 = evidential_regression_loss(gt_boxes[:,3], (pred_boxes[:,3],) + pred_dict['box_uncertainty'][3], self.evi_lambda)
                    evidential_loss_dim1 = evidential_regression_loss(gt_boxes[:,4], (pred_boxes[:,4],) + pred_dict['box_uncertainty'][4], self.evi_lambda)
                    evidential_loss_dim2 = evidential_regression_loss(gt_boxes[:,5], (pred_boxes[:,5],) + pred_dict['box_uncertainty'][5], self.evi_lambda)
                    evidential_loss = evidential_loss_loc0 + evidential_loss_loc1 + evidential_loss_loc2 + evidential_loss_dim0 + evidential_loss_dim1 + evidential_loss_dim2 
                else:
                    evidential_loss = evidential_regression_loss(gt_boxes[:,:6], (pred_boxes[:,:6],) + pred_dict['box_uncertainty'], self.evi_lambda)
                
                # rot_loss for dimloc
                if 'cos_sim' in self.yaw_loss:
                    cosine_sim = torch.cosine_similarity(pred_yaw, gt_yaw, dim=-1)
                    rot_loss = 1 - torch.mean(cosine_sim)    
                elif 'wrapped_angle_l1' in self.yaw_loss:
                    if 'pi' in self.yaw_loss[-1]:    
                        wrapped_pred_yaw = wrap_to_pi(pred_yaw)
                        wrapped_gt_yaw = wrap_to_pi(gt_yaw)
                    elif 'pi_over_2' in self.yaw_loss[-1]:
                        wrapped_pred_yaw = pred_yaw
                        wrapped_gt_yaw = wrap_to_minus_pi_half_pi_half(gt_yaw)
                    rot_loss = nn.SmoothL1Loss(reduction='none')(wrapped_pred_yaw, wrapped_gt_yaw).mean()
                else:
                    rot_loss = torch.tensor(0).cuda().float()

                loss_dict['rot_loss'] = (rot_loss.item(), num_gt_samples, 'losses')
                evidential_loss += rot_loss
            else:
                # Rot_loss
                if 'cos_sim' in self.yaw_loss:
                    cosine_sim = torch.cosine_similarity(pred_yaw, gt_yaw, dim=-1)
                    rot_loss = 1 - torch.mean(cosine_sim)
                    
                elif 'wrapped_angle_l1' in self.yaw_loss:
                    if 'pi' in self.yaw_loss[-1]:    
                        raise NotImplementedError
                    elif 'pi_over_2' in self.yaw_loss[-1]:
                        gt_boxes[:,6] = wrap_to_minus_pi_half_pi_half(gt_boxes[:,6])

                if self.use_unprocessed_gt:
                    gt_boxes_unprocessed = self.get_gt_unprocessed(gt_boxes)
                    pred_boxes_unprocessed = torch.cat([pred_dict['location_evi'], pred_dict['dimension_evi'], pred_dict['yaw_evi']], dim=-1)
                    gt_boxes_unprocessed[:,6] = wrap_to_minus_pi_half_pi_half(gt_boxes_unprocessed[:,6])
                    evidential_loss = evidential_regression_loss(gt_boxes_unprocessed, (pred_boxes_unprocessed,) + pred_dict['box_uncertainty'], self.evi_lambda)
                elif self.separate_heads:
                    evidential_loss_loc0 = evidential_regression_loss(gt_boxes[:,0], (pred_boxes[:,0],) + pred_dict['box_uncertainty'][0], self.evi_lambda)
                    evidential_loss_loc1 = evidential_regression_loss(gt_boxes[:,1], (pred_boxes[:,1],) + pred_dict['box_uncertainty'][1], self.evi_lambda)
                    evidential_loss_loc2 = evidential_regression_loss(gt_boxes[:,2], (pred_boxes[:,2],) + pred_dict['box_uncertainty'][2], self.evi_lambda)
                    evidential_loss_dim0 = evidential_regression_loss(gt_boxes[:,3], (pred_boxes[:,3],) + pred_dict['box_uncertainty'][3], self.evi_lambda)
                    evidential_loss_dim1 = evidential_regression_loss(gt_boxes[:,4], (pred_boxes[:,4],) + pred_dict['box_uncertainty'][4], self.evi_lambda)
                    evidential_loss_dim2 = evidential_regression_loss(gt_boxes[:,5], (pred_boxes[:,5],) + pred_dict['box_uncertainty'][5], self.evi_lambda)
                    evidential_loss_rot = evidential_regression_loss(gt_boxes[:,6], (pred_boxes[:,6],) + pred_dict['box_uncertainty'][6], self.evi_lambda)
                    evidential_loss = evidential_loss_loc0 + evidential_loss_loc1 + evidential_loss_loc2 + evidential_loss_dim0 + evidential_loss_dim1 + evidential_loss_dim2 + evidential_loss_rot
                else:
                    evidential_loss = evidential_regression_loss(gt_boxes, (pred_boxes,) + pred_dict['box_uncertainty'], self.evi_lambda, nll_weight = torch.tensor(self.nll_weight).float().cuda())

            if self.some_prints:
                pass
                    
            # Calculate epistemic and aleatoric uncertainty
            if self.separate_heads:
                aleatoric_unc = get_pred_unc_one_parameter(pred_dict['box_uncertainty'])
                epistemic_unc = get_pred_unc_one_parameter(pred_dict['box_uncertainty'], 'epistemic')
            else:
                aleatoric_unc = get_pred_evidential_aleatoric(pred_dict['box_uncertainty'], choose=self.choose_unc_idx)
                epistemic_unc = get_pred_evidential_epistemic(pred_dict['box_uncertainty'])

            # Calculate nu, alpha, beta
            if not self.separate_heads:
                v, alpha, beta = pred_dict['box_uncertainty']
            else:
                for par_idx in range(7):
                    v_par, alpha_par, beta_par = pred_dict['box_uncertainty'][par_idx]
                    try:
                        v = torch.cat((v, v_par), dim=1)
                        alpha = torch.cat((alpha, alpha_par), dim=1)
                        beta = torch.cat((beta, beta_par), dim=1)
                    except: 
                        v = v_par
                        alpha = alpha_par
                        beta = beta_par
      
            loss_dict['v'] = (torch.sum(v).item(), num_gt_samples, 'unc')
            loss_dict['alpha'] = (torch.sum(alpha).item(), num_gt_samples, 'unc')
            loss_dict['beta'] = (torch.sum(beta).item(), num_gt_samples, 'unc')

            min_corr_req = 2
            if iou3d.shape[0] >= min_corr_req:                                          # else error during val or train when batch size==1
                evi_iou3d_corr, _ = stats.pearsonr(iou3d.view(-1).cpu().detach().numpy(), aleatoric_unc.cpu().detach().numpy())   
                evi_iou3d_corr_epis, _ = stats.pearsonr(iou3d.view(-1).cpu().detach().numpy(), epistemic_unc.cpu().detach().numpy())      
                loss_dict['evi_iou_corr'] = (evi_iou3d_corr, num_gt_samples, 'iou')
                loss_dict['evi_iou_corr_epis'] = (evi_iou3d_corr_epis, num_gt_samples, 'iou')
            else:
                loss_dict['evi_iou_corr'] = (0, num_gt_samples, 'iou')
                loss_dict['evi_iou_corr_epis'] = (0, num_gt_samples, 'iou')

            # Option: High Uncertainty Regularization
            if self.high_unc_reg: 
                inner = pred_boxes[:,:6] - aleatoric_unc
                outer = pred_boxes[:,:6] - aleatoric_unc
                l_reg_un = torch.sqrt(torch.square(gt_boxes[:,:6] - pred_boxes[:,:6]))
                l_reg_un += torch.sqrt(torch.square(gt_boxes[:,:6] - inner))
                l_reg_un += torch.sqrt(torch.square(gt_boxes[:,:6] - outer))

            # Option: From EDL with Multitask Learning paper
            if self.l_mse:
                l_mse = torch.tensor(0).float().cuda()
                
                # Helbert version
                # U_v = ((beta * (v + 1)) / (alpha * v)).detach()
                # U_alpha = (((2*beta*(1+v))/v) * torch.exp(torch.digamma(alpha + 0.50) - torch.digamma(alpha)- 1)).detach()
                # thresh = {}
                # for col_indx in range(7):
                #     thresh[str(col_indx)] = torch.min(torch.min(U_v[:,col_indx], U_alpha[:,col_indx]).flatten()).detach()
                # sqrd_err = (gt_boxes - pred_boxes)**2
                
                # for col_indx in range(7):
                #     for idx in range(len(gt_boxes)):
                #         if sqrd_err[idx, col_indx] < thresh[str(col_indx)]: l_mse += sqrd_err[idx, col_indx]
                #         else: l_mse += 2 * torch.sqrt(thresh[str(col_indx)]).detach() * torch.abs(gt_boxes - pred_boxes)[idx, col_indx] - thresh[str(col_indx)].detach()
                
                # evidential_loss += l_mse

                for col_indx in range(7):
                    l_mse += modified_mse(pred_boxes[:,col_indx], v[:,col_indx], alpha[:,col_indx], beta[:,col_indx], gt_boxes[:,col_indx])

                evidential_loss += l_mse.mean()

            if self.some_prints:
                theta = (2 * v + alpha).mean(1)
                print(theta)

            if self.unc_guided_iou_loss:
                theta = (2 * v + alpha).mean(1)
                # theta_min = theta.min(0)[0].clone().detach()
                # theta_max = theta.max(0)[0].clone().detach()
                # evidence = ((theta - theta_min) / (theta_max - theta_min)) * (0.70 - 0.05) + 0.05
                # l_iou_guided = torch.log(evidence) * l_iou.reshape(-1) #+ torch.log(1-evidence) * (1-l_iou).reshape(-1)
                # l_iou_guided = l_iou_guided.mean()

            loss_dict['evidential_loss'] = (evidential_loss.item(), num_gt_samples, 'losses')

        loss_dict['loss_box']   = (loss_box.item(), num_gt_samples, 'losses')
        loss_dict['loss_conf']  = (loss_conf.item(), num_gt_samples, 'losses')

        # Option: Decouple IoU
        if self.decouple_iou: 
            loss_dict['loss_purity']  = (loss_purity.item(), num_gt_samples, 'losses')
            loss_dict['loss_integrity']  = (loss_integrity.item(), num_gt_samples, 'losses')

        # Save some loss info
        loss_dict['loss_dir']   = (loss_dir.item(), num_gt_samples, 'losses')
        loss_dict['iou3d']      = (iou3d.mean().item(), num_gt_samples, 'iou')
        loss_dict['iou2d']      = (iou2d.mean().item(), num_gt_samples, 'iou')
        loss_dict['err_loc']    = ((pred_loc - gt_loc).norm(dim=-1).mean().item(), num_gt_samples, 'box_err')
        loss_dict['err_dim']    = ((pred_dim - gt_dim).abs().mean().item(), num_gt_samples, 'box_err')
        loss_dict['err_yaw']    = (diff_yaw.abs().mean().item(), num_gt_samples, 'box_err')
        loss_dict['recall_7']   = ((iou3d>0.7).float().mean().item(), num_gt_samples, 'recall')
        loss_dict['recall_5']   = ((iou3d>0.5).float().mean().item(), num_gt_samples, 'recall')
        loss_dict['recall_3']   = ((iou3d>0.3).float().mean().item(), num_gt_samples, 'recall') 
        loss_dict['err_conf']   = (err_conf.item(), num_gt_samples, 'box_acc')
        loss_dict['acc_dir']    = (acc_dir.item(), num_gt_samples, 'box_acc')
        iou3d_histo             = iou3d.clone().detach().cpu()

        # Define the loss
        loss = torch.tensor(0).cuda().float()                       # loss_segment + loss_depth + loss_conf + loss_dir + (loss_box * self.box_loss_weight)                # original MTrans loss
        
        if self.evi_uncertainty and (self.unc_guided_loss or self.unc_guided_iou_loss):
            if self.rescale_unc:
                pass # there is an error
                # max_v = torch.clamp(v.max(), max=1) #torch.tensor(1).float().cuda()
                # min_v = torch.clamp(v.min(), min=0) #torch.tensor(0.60).float().cuda()
                # max_alpha = torch.clamp(alpha.max(), max=2) #torch.tensor(2).float().cuda()
                # min_alpha = torch.clamp(alpha.min(), min=1) #torch.tensor(1).float().cuda()
                # v_resc = (v - v.min(0)[0].detach()) / (v.max(0)[0].detach() - v.min(0)[0].detach()) * (max_v - min_v) + min_v
                # alpha_resc = (alpha - alpha.min(0)[0].detach()) / (alpha.max(0)[0].detach() - alpha.min(0)[0].detach()) * (max_alpha - min_alpha) + min_alpha
            else:
                v_resc = v.clone()
                alpha_resc = alpha.clone()

        if self.inc_lbox:
            if self.unc_guided_iou_loss: 
                loss += ((2 * v_resc + alpha_resc - 1).mean(1) * l_iou_unc.reshape(-1) - torch.log(2 * v_resc + alpha_resc - 1).mean(1)).mean() * self.box_loss_weight
            elif self.unc_guided_loss and not self.unc_guided_iou_loss:
                loss += ((2 * v_resc + alpha_resc - 1).mean(1) * l_iou_unc.reshape(-1) - torch.log(2 * v_resc + alpha_resc - 1).mean(1)).mean() * self.box_loss_weight
            else: 
                loss += (loss_box * self.box_loss_weight)
        if self.inc_lseg: 
            if self.unc_guided_loss: loss += ((2 * v_resc + alpha_resc - 1).mean(1) * (l_seg_unc+l_dice_unc).reshape(-1) - torch.log(2 * v_resc + alpha_resc - 1).mean(1)).mean() 
            else: loss += loss_segment
        if self.inc_ldepth: 
            if self.unc_guided_loss: loss += ((2 * v_resc + alpha_resc - 1).mean(1) * l_depth_unc.reshape(-1) - torch.log(2 * v_resc + alpha_resc - 1).mean(1)).mean() 
            else: loss += loss_depth
        if self.inc_lconf: 
            if self.unc_guided_loss: loss += ((2 * v_resc + alpha_resc - 1).mean(1) * l_conf_unc.reshape(-1) - torch.log(2 * v_resc + alpha_resc - 1).mean(1)).mean() 
            else: loss += loss_conf
        if self.inc_ldir: 
            if self.unc_guided_loss: loss += ((2 * v_resc + alpha_resc - 1).mean(1) * l_dir_unc.reshape(-1) - torch.log(2 * v_resc + alpha_resc - 1).mean(1)).mean() 
            else: loss += loss_dir
        if self.evi_uncertainty:
            loss += self.evi_loss_weight * evidential_loss  
        if self.decouple_iou:
            loss += loss_purity + loss_integrity
        if self.laplace_uncertainty and self.lapl_multi_unc:
            loss += l_loc0 + l_loc1 + l_loc2 + l_dim0 + l_dim1 + l_dim2 + l_yaw
            loss_dict['loss_loc_lapl']   = ((l_loc0 + l_loc1 + l_loc2).item(), num_gt_samples, 'losses')
            loss_dict['loss_dim_lapl']   = ((l_dim0 + l_dim1 + l_dim2).item(), num_gt_samples, 'losses')
            loss_dict['loss_yaw_lapl']   = (l_yaw.item(), num_gt_samples, 'losses')
        if self.ensemble:
            loss += self.ensemble_lambda * (l_loc0 + l_loc1 + l_loc2 + l_dim0 + l_dim1 + l_dim2 + l_yaw)
            loss_dict['loss_loc_ensemble']   = ((l_loc0 + l_loc1 + l_loc2).item(), num_gt_samples, 'losses')
            loss_dict['loss_dim_ensemble']   = ((l_dim0 + l_dim1 + l_dim2).item(), num_gt_samples, 'losses')
            loss_dict['loss_yaw_ensemble']   = (l_yaw.item(), num_gt_samples, 'losses')
        
        loss_dict['loss'] = (loss.item(), B, 'loss')

        if self.evi_uncertainty:
            return loss_dict, loss, iou3d_histo, loss_box, iou3d.view(-1).cpu().detach().numpy(),  aleatoric_unc.cpu().detach().numpy(), epistemic_unc.cpu().detach().numpy(), \
                v.cpu().detach().numpy(), alpha.cpu().detach().numpy(), beta.cpu().detach().numpy(), gt_boxes.clone().cpu().detach().numpy(), pred_boxes.clone().cpu().detach().numpy()                                       # loss_dict, loss, loss_segment, loss_depth, loss_box, iou3d_histo
        elif self.ensemble:
            return loss_dict, loss, iou3d_histo, loss_box, iou3d.view(-1).cpu().detach().numpy(), gt_boxes.clone().cpu().detach().numpy(), pred_boxes.clone().cpu().detach().numpy(), var.clone().cpu().detach().numpy()
        elif self.mcdo:
            return loss_dict, loss, iou3d_histo, loss_box, iou3d.view(-1).cpu().detach().numpy(), gt_boxes.clone().cpu().detach().numpy(), pred_boxes.clone().cpu().detach().numpy()
        else:
            return loss_dict, loss, iou3d_histo, loss_box, iou3d.view(-1).cpu().detach().numpy(), gt_boxes.clone().cpu().detach().numpy(), pred_boxes.clone().cpu().detach().numpy()   

    def get_gt_unprocessed(self, gt_boxes):
        device = gt_boxes.device
        location, dimension, yaw = gt_boxes[:,:3], gt_boxes[:,3:6], gt_boxes[:,6]
        dim_anchor = torch.tensor(self.cfgs.anchor, device=device).view(1, 3)       # 1 x 3
        da = torch.norm(dim_anchor[:, :2], dim=-1, keepdim=True)                    # 1 x 1 # da = sqrt(3.9**2 + 1.6**2)
        ha = dim_anchor[:, 2:3]                                                     # 1 x 1
        dim_unprocessed = torch.log(dimension / dim_anchor)
        loc_unprocessed = location / torch.cat([da, da, ha], dim=-1)  
        
        # pred_loc = location * torch.cat([da, da, ha], dim=-1)                   # B x 3 # TODO What is the intuition for this?
        # pred_dim = torch.exp(dimension) * dim_anchor                                # B x 3 # TODO Why exp? Intuition?     

        return torch.cat([loc_unprocessed, dim_unprocessed, yaw.view(-1,1)], dim=-1)                            

    def clamp_orientation_range(self, angles):
        # angles: (B, 1)
        a = angles.clone()          # Bl x 1
        for i in range(a.size(0)):  # Angle should fall between -np.pi and np.pi
            while a[i] > np.pi:
                a[i] = a[i] - np.pi * 2
            while a[i] <= -np.pi:
                a[i] = a[i] + np.pi*2
        assert (a<=np.pi).all() and (a>=-np.pi).all()
        return a

    def adjust_direction(self, yaw, dir):
        # yaw: (B, 1), dir: (B, 1) - long
        yaw = self.clamp_orientation_range(yaw)
        for i in range(yaw.size(0)):
            # check direction
            if dir[i]==1 and not (yaw[i]>=-np.pi/2 and yaw[i]<np.pi/2):
                    yaw[i] = yaw[i] + np.pi
            elif dir[i]==0 and (yaw[i]>=-np.pi/2 and yaw[i]<np.pi/2):
                    yaw[i] = yaw[i] + np.pi
        return yaw

    def format_kitti_labels(self, pred_dict, data_dict, with_score=True):
        location, dimension, yaw = pred_dict['location'], pred_dict['dimension'], pred_dict['yaw']
        location = location + pred_dict['subcloud_center'] + pred_dict['second_offset']
        direction = pred_dict['direction'].argmax(dim=-1)
        yaw = self.adjust_direction(yaw, direction)
        labels = []
        
        for i in range(pred_dict['batch_size']):
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

            if 'scores' in data_dict.keys():
                # for test result, MAPGen confidence * 2D Box score
                score = score * data_dict['scores'][i] / max(pred_dict['conf']).item()

            if with_score:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f} {score:.4f}')
            else:
                labels.append(f'{data_dict.class_names[i]} {truncated:.2f} {occluded} {alpha:.2f} {box_2d} {dim} {loc} {a:.2f}')
        return labels, data_dict.frames
