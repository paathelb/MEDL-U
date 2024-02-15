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

import torch.special as sp

from easydict import EasyDict

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*4)

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        min_val = 1e-6
        
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, torch.nn.Softplus()(logv) + min_val, torch.nn.Softplus()(logalpha) + min_val + 1, torch.nn.Softplus()(logbeta) + min_val

def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        
    return nll

def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi

def evidential_regression_loss(y, pred, coeff=1.0, weight_loss = None):
    gamma, v, alpha, beta = pred
    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    
    if weight_loss is None:
        loss_ = loss_nll.mean() + coeff * (loss_reg.mean() - 1e-4)
        return loss_
    else:
        weight_loss = torch.tensor(weight_loss).view(-1,1).cuda()
        loss_ = torch.sum(weight_loss * loss_nll) + coeff * (torch.sum(loss_reg * weight_loss) - 1e-4)
        
        return loss_
    
class UncertaintyHead(torch.nn.Module):
    def __init__(self, hidden_channels, uncertainty, pred_size):
        super(UncertaintyHead, self).__init__()
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        
        self.uncertainty = uncertainty
        
        if self.uncertainty == 'False':
            self.lin2 = nn.Linear(hidden_channels // 2, pred_size)
        elif self.uncertainty == 'evidential':
            self.lin2 = LinearNormalGamma(hidden_channels //2, pred_size)
        # elif self.uncertainty == 'gaussian':
        #     self.lin2 = LinearNormal(hidden_channels // 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        if self.uncertainty == 'False':
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            self.lin2.bias.data.fill_(0)

    def forward(self, v, batch=None):
        v = self.lin1(v)
        v = self.act(v)
        
        if self.uncertainty == 'False':
            v = self.lin2(v)
            #u = scatter(v, batch, dim=0)
            u = v
        elif self.uncertainty in ['evidential', 'gaussian']:
            #u = scatter(v, batch, dim=0)
            u = v
            u = self.lin2(u)
        return u








class MTrans(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.parameters_loaded = []             # record the names of parameters loaded from previous stage
        self.evi_uncertainty = cfgs.evi_uncertainty
        self.decouple_iou = cfgs.decouple_iou
        self.laplace_uncertainty = cfgs.laplace_uncertainty

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
                self.box_head_1 = nn.Sequential(
                    nn.Linear(hidden_size+cimg+hidden_size, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=cfgs.box_drop),
                    #nn.Linear(512, 7)
                )
                self.box_head_evi = UncertaintyHead(512, 'evidential', 7)
                
                # self.conf_dir_head_1 = nn.Sequential(
                #     nn.Linear(hidden_size+cimg+hidden_size, 512),
                #     nn.LayerNorm(512),
                #     nn.ReLU(inplace=True),
                #     nn.Dropout(p=0.4),
                #     #nn.Linear(512, 3)
                # )
                # self.conf_dir_head_evi = UncertaintyHead(512, 'evidential', 3)

            if self.laplace_uncertainty:
                self.lapl_unc_head = nn.Sequential(
                    nn.Linear(hidden_size+cimg+hidden_size, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4),
                    nn.Linear(512, 8)
                )
                
            else: 
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
        image_features, _ = self.cnn(image)       # Deep Neural Network of Conv2d, BatchNorm2d, and ReLU      # B X 512 X H X W
        
        # -------------------------------------------------------- 2. Build new cloud, which contains blank slots to be interpolated -----------------------
        B, N, _ = sub_cloud.size()
        scale = self.cfgs.sparse_query_rate
        qH, qW = H//scale, W//scale             # TODO Intuition?

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

        # NOTE Box head and conf dir head have exactly the same architecture
        # TODO Modify to include neural network for evidential regression & uncertainty estimation
        
        box_feature = torch.cat([cls_f3d.squeeze(1), global_f3d, global_f2d], dim=-1)                   # B x 2048

        if self.evi_uncertainty:
            box_feature_2 = self.box_head_1(box_feature)                                                # B x 7
            boxes = self.box_head_evi(box_feature_2)                                                    # Tuple of length 4: each is a tensor of shape 4 x 7
            evi_uncertainty_values = boxes[1:]
            box = boxes[0]                                                                              # B x 7
            pred_dict['evi_uncertainty_values'] = evi_uncertainty_values 

            # conf_dir_pred_feature = self.conf_dir_head_1(box_feature)                                 # B x 3
            # conf_dir_preds = self.conf_dir_head_evi(conf_dir_pred_feature)
            # conf_dir_preds_uncertainty = conf_dir_preds[1:]
            # conf_dir_pred = conf_dir_preds[0]
            # pred_dict['conf_dir_preds_uncertainty'] = conf_dir_preds_uncertainty

            location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]                            # B x 3 # B x 3 # B x 1
        elif self.laplace_uncertainty:
            box = self.lapl_unc_head(box_feature)  
            location, dimension, yaw, lapl_unc = box[:, 0:3], box[:, 3:6], box[:, 6:7], box[:, 7:8]                      # B x 3 # B x 3 # B x 1  
            pred_dict['lapl_unc'] = lapl_unc
        else:
            box = self.box_head(box_feature)     
            location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]                                        # B x 3 # B x 3 # B x 1  
            
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






        # Post-processing
        dim_anchor = torch.tensor(self.cfgs.anchor, device=device).view(1, 3)       # 1 x 3
        da = torch.norm(dim_anchor[:, :2], dim=-1, keepdim=True)                    # 1 x 1 # da = sqrt(3.9**2 + 1.6**2)
        ha = dim_anchor[:, 2:3]                                                     # 1 x 1
        pred_loc = location * torch.cat([da, da, ha], dim=-1)                       # B x 3 # TODO What is the intuition for this?
        pred_dim = torch.exp(dimension) * dim_anchor                                # B x 3 # TODO Why exp? Intuition?
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

        # TODO Verify if correct
        if weights is not None:
            if all(elem==1 for elem in weights):
                weights = None

        loss_dict = {}
        B = pred_dict['batch_size'].sum().item()
        has_label = data_dict['use_3d_label']                      # (B)
        real_point_mask = data_dict['real_point_mask']             # (B, N)




        # ----------------------------------------------------------------------------- 1. Foreground loss ----------------------------------------------------------------------------- 
        segment_logits = pred_dict['pred_foreground_logits'].transpose(-1,-2)           # (B, 2, N)     # For foreground and background
        gt_segment_label = data_dict['foreground_label']                                # (B, N)        # How did we get data_dict.foreground_label?

        if self.evi_uncertainty:
            gt_segment_label_scatter = torch.zeros(gt_segment_label.shape[0], 2, gt_segment_label[1])
            gt_segment_label_scatter.scatter_(1, gt_segment_label.unsqueeze(1), 1)
            alpha = F.softplus(segment_logits) + 1
            S = alpha.sum(dim=1).unsqueeze(1).repeat(1,2,1)        # B x 2 x N
            L_theta = torch.mul(gt_segment_label_scatter, (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)

            C = 2
            alpha_tld = gt_segment_label_scatter + (1-gt_segment_label_scatter) * alpha
            S_tld = alpha.sum(dim=1).unsqueeze(1).repeat(1,2,1)
            loss_KL = torch.log(torch.lgamma(alpha_tld.sum(dim=1)) / (torch.lgamma(C) * torch.lgamma(alpha_tld).prod(dim=1))) \
                + ((alpha_tld - 1) * (torch.digamma(alpha_tld) - torch.digamma(S_tld)).sum(dim=1))
        
        # Loss only for those have 3D label
        segment_gt, segment_logits = gt_segment_label[has_label], segment_logits[has_label]    # Bl x N   # Bl x 2 x N
        criterion_segment = nn.CrossEntropyLoss(reduction='none', ignore_index=2)              # Ignore label of 2
        loss_segment = criterion_segment(segment_logits, segment_gt)                           # Bl x N    # TODO Why segment_logits does not sum up to 1?
        
        # Balance foreground and background. Take the mean across batch samples
        lseg = 0
        if (segment_gt==1).sum() > 0:
            lseg = lseg + (loss_segment * (segment_gt==1)).sum(dim=-1) / ((segment_gt==1).sum(dim=-1) + 1e-6) # Bl # Average segment_loss for foreground
        if (segment_gt==0).sum() > 0:
            lseg = lseg + (loss_segment * (segment_gt==0)).sum(dim=-1) / ((segment_gt==0).sum(dim=-1) + 1e-6) # Bl # Average segment_loss for background

        if weights is None:
            loss_segment = lseg.mean()          
        else:
            loss_segment = self.get_weighted_loss(lseg, weights)

        # Add Dice Loss to Loss Segment         # TODO What is dice loss?
        segment_prob = segment_logits.softmax(dim=1)[:, 1, :]                                           # Bl x N        # TODO Probability of being classified as foreground?
        inter = 2 * (segment_prob * (segment_gt==1)).sum(dim=-1) + 1e-6                                 # Bl            # TODO Intuition? 
        uni = (segment_prob * (segment_gt != 2)).sum(dim=-1) + (segment_gt == 1).sum(dim=-1) + 1e-6     # Bl            # TODO Intuition? 
        dice_loss = 1 - inter/uni                                                                       # Bl            # TODO Intuition? 

        if weights is None:
            dice_loss = dice_loss.mean()          
        else:
            dice_loss = self.get_weighted_loss(dice_loss, weights)

        loss_segment = loss_segment + dice_loss                         

        # Metric: Segment IoU
        segment_pred = segment_logits.argmax(dim=1) * (segment_gt != 2)             # Bl x N # TODO Why automatically classify mask as background?
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

        if self.laplace_uncertainty:
            log_variance = pred_dict['lapl_unc'] #.clamp(min=-10, max=10)
            l_iou =  1.4142 * torch.exp(-log_variance) * l_iou + log_variance

        if weights is None:
            loss_box = l_iou.mean()          
        else:
            loss_box = self.get_weighted_loss(l_iou, weights)




        # Loss for direction/ yaw angle     # NOTE It did not use the predicted yaw
        gt_dir = self.clamp_orientation_range(gt_yaw)                                                   # Bl x 1 # TODO What is the intuition in doing this? 
        gt_dir = ((gt_dir >= -np.pi/2) * (gt_dir < np.pi/2)).long().squeeze(-1)                         # Bl
        pred_dir = pred_dict['direction'][has_label]  
        criterion_dir = torch.nn.CrossEntropyLoss(reduction='none')                                     # Bl x 2
        loss_dir = criterion_dir(pred_dir, gt_dir)          

        if weights is None:
            loss_dir = loss_dir.mean()   
        else:
            loss_dir = self.get_weighted_loss(loss_dir, weights)

        acc_dir = (pred_dir.argmax(dim=-1) == gt_dir).sum() / num_gt_samples
        


        # Loss for confidence
        # TODO Weighted version
        if self.decouple_iou:
            pred_purity = pred_dict['purity'][has_label] 
            pred_integrity = pred_dict['integrity'][has_label]  
            confidence = pred_dict['conf'][has_label]  
            criterion_conf = torch.nn.SmoothL1Loss(reduction='none')
            loss_purity = criterion_conf(pred_integrity, integrity.squeeze()).mean()          #-(purity.squeeze() * torch.log(pred_purity) + (1-purity).squeeze() * torch.log(1-pred_purity)).mean()
            loss_integrity = criterion_conf(pred_purity, purity.squeeze()).mean()      #-(integrity.squeeze() * torch.log(pred_integrity) + (1-integrity).squeeze() * torch.log(1-pred_integrity)).mean()
            loss_conf = criterion_conf(confidence, iou3d.squeeze())               #-(iou3d.squeeze() * torch.log(confidence) + (1-iou3d) * torch.log(1-confidence))
        else:
            confidence = pred_dict['conf'][has_label]  
            criterion_conf = torch.nn.SmoothL1Loss(reduction='none')                                        # Bl x 1
            loss_conf = criterion_conf(confidence, iou3d)                                                   # Single value

        if weights is None:
            loss_conf = loss_conf.mean()   
        else:
            loss_conf = self.get_weighted_loss(loss_conf, weights)
                                                               
        err_conf = ((confidence - iou3d).abs().sum() / num_gt_samples)                                  # Single value # TODO Why define it this way?
        assert not iou3d.isnan().any()
        


        #  --------------------------------------------------------------- 4. Evidential Regression Loss -------------------------------------------------------------------------
        if self.evi_uncertainty:
            evi_lambda = 1
            evidential_loss = evidential_regression_loss(gt_boxes, (pred_boxes,) + pred_dict['evi_uncertainty_values'], evi_lambda)

            # Loss on high uncertainty dim predictions (surpassing a threshold)
            uncertainty = self.get_pred_evidential_epistemic(pred_dict['evi_uncertainty_values'])
            inner = pred_boxes - uncertainty
            outer = pred_boxes + uncertainty
            evi_reg_loss = torch.sqrt((gt_boxes-pred_boxes)**2) + torch.sqrt((gt_boxes-inner)**2) + torch.sqrt((gt_boxes-outer)**2)
            loss_dict['evidential_loss'] = (evidential_loss.item(), num_gt_samples, 'losses')

        loss_dict['loss_box']   = (loss_box.item(), num_gt_samples, 'losses')
        loss_dict['loss_conf']  = (loss_conf.item(), num_gt_samples, 'losses')

        if self.decouple_iou: 
            loss_dict['loss_purity']  = (loss_purity.item(), num_gt_samples, 'losses')
            loss_dict['loss_integrity']  = (loss_integrity.item(), num_gt_samples, 'losses')

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

        loss = loss_segment + loss_depth + (loss_box * 5) + loss_conf + loss_dir

        if self.evi_uncertainty:
            loss += evidential_loss  

        if self.decouple_iou:
            loss += loss_purity + loss_integrity
        
        loss_dict['loss'] = (loss.item(), B, 'loss')

        return loss_dict, loss, iou3d_histo, loss_box                                                 # loss_dict, loss, loss_segment, loss_depth, loss_box, iou3d_histo

    def get_pred_evidential_epistemic(self, out):
        v, alpha, beta = out
        var = beta / (v * (alpha - 1))
        return torch.mean(var, dim=1)

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
