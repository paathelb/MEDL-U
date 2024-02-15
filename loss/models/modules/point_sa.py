"""
    Apply self-attention across the points
"""
import torch
from torch import nn
import math
import numpy as np
from utils.point_ops import get_knn_values

def cal_sinusoid_encoding(dims, coords):
    # coords: (B, N)
    position_embedding_channel = dims
    device = coords.device
    hidden = torch.arange(position_embedding_channel, device=device)        # dims
    hidden = torch.div(hidden, 2, rounding_mode='floor') * 2 / position_embedding_channel       # dims # TODO Intuition? Number increasing from 0 to close to 1
    hidden = torch.pow(10000, hidden)       # dims
    coords = coords.unsqueeze(-1) / hidden.view(1, 1, -1)    # (B, N, dims)
    coords[:, :, 0::2] = torch.sin(coords[:, :, 0::2])       # (B, N, dims/2) Update half
    coords[:, :, 1::2] = torch.cos(coords[:, :, 1::2])       # (B, N, dims/2) Update other half
    return coords # (B, N, dims)

class PointSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head_size = cfg.hidden_size // cfg.num_heads

        # self.key = nn.Linear(cfg.position_embedding_channel+cfg.input_pts_channel + cfg.input_img_channel, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        
        # self.value = nn.Linear(cfg.position_embedding_channel+cfg.input_pts_channel + cfg.input_img_channel, cfg.hidden_size)
        
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.layerNorm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, query_states, key_states, attn_mask):
        """
            Args: 
                query_states: (B, N+M, Ci) float
                key_states: (B, N, Ci) float
                attn_mask: (B, N)
        """
        values = self.value(key_states)     # B x N x 768
        query = self.query(query_states)    # B x N or M x 768
        key = self.key(key_states)          # B x N x 768

        query, key, values = self.split_heads(query), self.split_heads(key), self.split_heads(values)   # (B, num_heads, N or M, head_size) # Just reshaping # num_heads*head_size = 768

        attention_scores = torch.matmul(query, key.transpose(-1, -2))       # (B, num_heads, N or M, N)
        if attn_mask is not None:
            attention_scores = attention_scores - ((~attn_mask).float()*1e10).unsqueeze(1).unsqueeze(2) # B x num_heads x (N or M) x N # Giving very low attention scores to padding and mask

        attention_scores = nn.Softmax(dim=-1)(attention_scores / math.sqrt(self.head_size)) # B x num_heads x (N or M) x (N or N+1)
        
        ### For visualization ###
        # if data is not None:
        #    debug=1
            # i = 1
            # fig, ax = plt.subplots()
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            # ax.imshow(data.images[i].permute(1, 2, 0).detach().cpu())
            # pts_2d = data.sub_clouds2d[i][(data.real_point_mask==1)[i]]
            # scores = attention_scores[i, :, :, (data.real_point_mask==1)[i]]    # (numhead, numK, num2d)
            # scores = scores.mean(dim=1).mean(dim=0)
            # im = ax.scatter(pts_2d[:, 0].cpu(), pts_2d[:, 1].cpu(), c=scores, cmap='plasma', s=25)
            # fig.colorbar(im, ax=ax, orientation='vertical')
            # ax.set_xlim([0,111])
            # ax.set_ylim([111,0])

            # i = 1
            # pts_3d = data.sub_clouds[i][(data.real_point_mask==1)[i]]
            # box = torch.cat([data.locations, data.dimensions, data.yaws], dim=-1)[i].cpu()
            # scores = attention_scores[i, :, :, (data.real_point_mask==1)[i]]    # (numhead, numK, num2d)
            # scores = scores.mean(dim=1).mean(dim=0)
            # scores = (scores - scores.min()) / (scores.max()-scores.min())
            # colors = plt.cm.plasma(scores.cpu().numpy())*255
            # colors = (colors[:, :3]).astype(np.uint8)
            # fig = visualize_point_cloud(pts_3d, color=colors, pred_3dbox=box, point_size=0.1)

            # display_plotting(0, elevation=75, focalpoint=(-3, -1, 0), distance=10, azimuth=-179, roll=90)         #f1
            # display_plotting(0, elevation=77, focalpoint=(-3, -1, 0), distance=15, azimuth=-200, roll=95)         #f2
            # display_plotting(0, elevation=75, focalpoint=(-3, -1, 0), distance=15, azimuth=-159, roll=85)         #f4
            # display_plotting(0, elevation=75, focalpoint=(-3, -1, 0), distance=10, azimuth=-159, roll=85)         #f4_2
            # display_plotting(0, elevation=75, focalpoint=(-3, 0, 0), distance=18, azimuth=-159, roll=85)          #f6
            # display_plotting(0, elevation=75, focalpoint=(-3, -1, 0), distance=20, azimuth=-159, roll=85)         #f6_2
            # display_plotting(0, elevation=70, focalpoint=(-3, -1.5, 0), distance=15, azimuth=-159, roll=82)       #f6_3
            
        ########################

        attention_scores = self.dropout(attention_scores)       # B x num_heads x (N or M) x (N or N+1) # TODO Intuition with dropout layer
        values = torch.matmul(attention_scores, values)                 # (B, num_heads, (N or M), head_size) # TODO Intuition?
        values = self.merge_heads(values)                               # (B, N or M, hidden_size) # Just permuting and reshaping

        values = self.dropout(self.dense(values))                       # (B, N or M, hidden_size)
        values = self.layerNorm(values + query_states)                  # (B, N or M, hidden_size) # Why add values to query_states?
        return values                                                   # (B, N or M, hidden_size)

    def split_heads(self, values):
        # values: (B, N, C)
        num_heads = self.cfg.num_heads
        B, N, C = values.size()
        values = values.view(B, N, num_heads, -1).permute(0, 2, 1, 3)         # (B, num_heads, N, head_size)
        return values

    def merge_heads(self, values):
        # values: (B, num_heads, N, head_size)
        B, _, N, _ = values.size()
        values = values.permute(0, 2, 1, 3).reshape(B, N, -1)
        return values

class PointAttentionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = PointSelfAttention(cfg)
        self.dense1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.act = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.layernorm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, query_states, key_states, attn_mask):
        new_feature_3d = self.attention(query_states, key_states, attn_mask)     # B x N or M x hidden_size
        intermediate = self.act(self.dense1(new_feature_3d))                                            # B x N or M x 1024
        intermediate = self.dropout(self.dense2(intermediate))                                          # B x N or M x hidden_size
        output = self.layernorm(intermediate + new_feature_3d)                                          # B x N or M x hidden_size
        return output                                                                                   # B x N or M x hidden_size

class AttentionPointEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_layers = cfg.num_layers
        layers = [PointAttentionLayer(cfg) for i in range(num_layers)]
        self.layers = nn.ModuleList(layers)

        # fusion method
        if cfg.fuse_method=='ADD':
            self.ln1 = nn.LayerNorm(cfg.input_pts_channel)
            self.down_channel = nn.Sequential(
                nn.Linear(cfg.input_pts_channel, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size)
            )
        elif cfg.fuse_method=='CAT':
            self.down_channel = nn.Sequential(
                nn.Linear(cfg.input_pts_channel*3, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size)
            )
        elif cfg.fuse_method=='GATE':
            self.gating = nn.Sequential(
                nn.Linear(cfg.input_pts_channel*3, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Softmax(dim=-1)
            )
            self.down_channel = nn.Linear(cfg.input_pts_channel, cfg.hidden_size)

        if cfg.fore_attn:
            self.fore_attn = nn.Sequential(
                nn.Linear(cfg.hidden_size, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
                )

        if cfg.use_cls_token:
            self.cls_f3d = nn.Parameter(torch.zeros(cfg.hidden_size))
            self.cls_f3d = nn.init.normal_(self.cls_f3d)    
    
    def forward(self, query_c2d, query_f2d, query_f3d, key_c2d, key_f2d, key_f3d, attn_mask):
        """
            Args: 
                query_c2d: (B, M, 2) float
                query_f2d: (B, M, Ci) float
                query_f3d: (B, M, Cp) float or None
                key_c2d:    (B, N, 2) 
                key_f2d:    (B, N, Ci)
                key_f3d:    (B, N, Cp)
                attn_mask:  (B, N)
        """
        B, N, _ = key_c2d.size()
        query_c2d = torch.cat([cal_sinusoid_encoding(query_f2d.size(-1)//2, query_c2d[:, :, 0]),
                               cal_sinusoid_encoding(query_f2d.size(-1)//2, query_c2d[:, :, 1]),], dim=-1) # B x qH*qW x Ci # TODO Why divide the number of features by 2?
        key_c2d = torch.cat([cal_sinusoid_encoding(key_f2d.size(-1)//2, key_c2d[:, :, 0]),
                             cal_sinusoid_encoding(key_f2d.size(-1)//2, key_c2d[:, :, 1]),], dim=-1) # B x N x Ci   # TODO Intuition behind sinusoid_encoding on 2D? Why on 2D?

        if self.cfg.fuse_method=='ADD':
            query_states = self.down_channel(self.ln1(query_f2d + query_f3d + query_c2d))
            key_states = self.down_channel(self.ln1(key_f2d + key_f3d + key_c2d))
        elif self.cfg.fuse_method=='CAT':
            query_states = self.down_channel(torch.cat([query_f2d, query_f3d, query_c2d], dim=-1))  # B x qH*qW x 768 # Before passing: B x qH*qW x 3*Ci
            key_states = self.down_channel(torch.cat([key_f2d, key_f3d, key_c2d], dim=-1))          # B x N x 768     # Before passing: B x N x 3*Ci
        elif self.cfg.fuse_method=='GATE':
            query_weights = self.gating(torch.cat([query_f2d, query_f3d, query_c2d], dim=-1))
            key_weights = self.gating(torch.cat([key_f2d, key_f3d, key_c2d], dim=-1))
            query_states = torch.matmul(query_weights.unsqueeze(-2), torch.stack([query_f2d, query_f3d, query_c2d], dim=2))
            key_states = torch.matmul(key_weights.unsqueeze(-2), torch.stack([key_f2d, key_f3d, key_c2d], dim=2))
            query_states = self.down_channel(query_states.squeeze(-2))
            key_states = self.down_channel(key_states.squeeze(-2))

        cls_f3d = self.cls_f3d.view(1, 1, -1).repeat(B, 1, 1) # B x 1 x 768 # I guess self.cls_f3d is normally distributed with mean 0 and std 1
        
        for l in self.layers:
            res = l(torch.cat([key_states, query_states], dim=1), key_states, attn_mask)
            key_states, query_states = res[:, :N, :], res[:, N:, :] # B x N x hidden_size # B x M x hidden_size
            if self.cfg.fore_attn:
                 res = self.fore_attn(res) * res
                 
            cls_f3d = l(cls_f3d, torch.cat([cls_f3d, res], dim=1), None)

        return query_states, key_states, cls_f3d
        