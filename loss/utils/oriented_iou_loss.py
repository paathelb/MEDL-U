import torch
from .min_enclosing_box import smallest_bounding_box
from .box_intersection_2d import oriented_box_intersection_2d

def box2corners_th(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]

    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)

    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)

    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)

    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

def cal_iou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1) # B x N x 4 x 2
    corners2 = box2corners_th(box2) # B x N x 4 x 2

    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3] # Height x Width # B x N
    area2 = box2[:, :, 2] * box2[:, :, 3] # B x N
    u = area1 + area2 - inter_area # TODO Intuition?
    iou = inter_area / u # B x N
    return iou, corners1, corners2, u

def cal_iou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, decouple_iou=False, verbose=False):
    """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,l,w,h,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,l,w,h,alpha)
    """
    box1 = box3d1[..., [0,1,3,4,6]] # B x N x 5
    box2 = box3d2[..., [0,1,3,4,6]] # B x N x 5

    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5 # B x 1 # Why should we add/subtract the z to the length*0.50?
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5 # B x 1
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5 # B x 1
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5 # B x 1

    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.) # B x 1 # TODO Intuition
    iou_2d, corners1, corners2, u = cal_iou(box1, box2)        # (B, N), B x N x 4 x 2, B x N x 4 x 2, B x N
    intersection_3d = iou_2d * u * z_overlap # TODO Intuition? # intersection times z_overlap?

    v1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5] # B x N   # Volume
    v2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5] # B x N
    u3d = v1 + v2 - intersection_3d # B x N # 3D union

    # assert torch.all(box3d1[:,:,3:6]>=0)
    
    if verbose:
        z_range = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.) # B x N # Distance between zmax and zmin
        if not decouple_iou:
            return intersection_3d / u3d, corners1, corners2, z_range, u3d, iou_2d
        else:
            return intersection_3d / u3d, intersection_3d/v1, intersection_3d/v2, corners1, corners2, z_range, u3d, iou_2d
    else:
        return intersection_3d / u3d

def cal_giou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, enclosing_type:str="smallest"):
    """calculated 3d GIoU loss. assume the 3d bounding boxes are only rotated around z axis

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".

    Returns:
        (torch.Tensor): (B, N) 3d GIoU loss
        (torch.Tensor): (B, N) 3d IoU
    """
    assert torch.all(box3d1[:,:,3:6]>=0)
    iou_3d, corners1, corners2, z_range, u3d,iou_2d = cal_iou_3d(box3d1, box3d2, verbose=True)
    w, h = enclosing_box(corners1, corners2, enclosing_type)
    v_c = z_range * w * h
    # giou_loss = 1. - iou3d + (v_c - u3d)/v_c
    giou_loss = 1-iou_3d
    return giou_loss, iou_3d, iou_2d

def cal_diou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, decouple_iou=False, enclosing_type:str="smallest"):
    """calculated 3d DIoU loss. assume the 3d bounding boxes are only rotated around z axis

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".

    Returns:
        (torch.Tensor): (B, N) 3d DIoU loss
        (torch.Tensor): (B, N) 3d IoU
    """
    if not decouple_iou:
        iou_3d, corners1, corners2, z_range, u3d, iou_2d = cal_iou_3d(box3d1, box3d2, verbose=True)         # B x N  # B x N x 4 x 2  # B x N x 4 x 2  # B x N  # B x N  # B x N
    else:
        iou_3d, purity, integrity, corners1, corners2, z_range, u3d, iou_2d = cal_iou_3d(box3d1, box3d2, decouple_iou=True, verbose=True)

    w, h = enclosing_box(corners1, corners2, enclosing_type)            # B x N  # B x N        # TODO Is this type of enclosing?
    x_offset = box3d1[...,0] - box3d2[..., 0]                           # Bl x N
    y_offset = box3d1[...,1] - box3d2[..., 1]                           # Bl x N
    z_offset = box3d1[...,2] - box3d2[..., 2]                           # Bl x N
    d2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset      # Bl x N        # TODO Intuition? 
    c2 = w*w + h*h + z_range*z_range                                    # Bl x N
    diou = 1. - iou_3d + d2/c2                                          # Bl x N        # TODO Intuition?

    if not decouple_iou:
        return diou, iou_3d, iou_2d
    else:
        return diou, iou_3d, purity, integrity, iou_2d 

def enclosing_box(corners1:torch.Tensor, corners2:torch.Tensor, enclosing_type:str="smallest"):
    if enclosing_type == "aligned":
        return enclosing_box_aligned(corners1, corners2)
    elif enclosing_type == "pca":
        return enclosing_box_pca(corners1, corners2)
    elif enclosing_type == "smallest":
        return smallest_bounding_box(torch.cat([corners1, corners2], dim=-2))
    else:
        ValueError("Unknow type enclosing. Supported: aligned, pca, smallest")

def enclosing_box_aligned(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the smallest enclosing box (axis-aligned)

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    w = x_max - x_min       # (B, N)
    h = y_max - y_min
    return w, h

def enclosing_box_pca(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate the rotated smallest enclosing box using PCA

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
    
    Returns:
        w (torch.Tensor): (B, N)
        h (torch.Tensor): (B, N)
    """
    B = corners1.size()[0]
    c = torch.cat([corners1, corners2], dim=2)      # (B, N, 8, 2)
    c = c - torch.mean(c, dim=2, keepdim=True)      # normalization
    c = c.view([-1, 8, 2])                          # (B*N, 8, 2)
    ct = c.transpose(1, 2)                          # (B*N, 2, 8)
    ctc = torch.bmm(ct, c)                          # (B*N, 2, 2)
    # NOTE: the build in symeig is slow!
    # _, v = ctc.symeig(eigenvectors=True)
    # v1 = v[:, 0, :].unsqueeze(1)                   
    # v2 = v[:, 1, :].unsqueeze(1)
    v1, v2 = eigenvector_22(ctc)
    v1 = v1.unsqueeze(1)                            # (B*N, 1, 2), eigen value
    v2 = v2.unsqueeze(1)
    p1 = torch.sum(c * v1, dim=-1)                  # (B*N, 8), first principle component
    p2 = torch.sum(c * v2, dim=-1)                  # (B*N, 8), second principle component
    w = p1.max(dim=-1)[0] - p1.min(dim=-1)[0]       # (B*N, ),  width of rotated enclosing box
    h = p2.max(dim=-1)[0] - p2.min(dim=-1)[0]       # (B*N, ),  height of rotated enclosing box
    return w.view([B, -1]), h.view([B, -1])

def eigenvector_22(x:torch.Tensor):
    """return eigenvector of 2x2 symmetric matrix using closed form
    
    https://math.stackexchange.com/questions/8672/eigenvalues-and-eigenvectors-of-2-times-2-matrix
    
    The calculation is done by using double precision

    Args:
        x (torch.Tensor): (..., 2, 2), symmetric, semi-definite
    
    Return:
        v1 (torch.Tensor): (..., 2)
        v2 (torch.Tensor): (..., 2)
    """
    # NOTE: must use doule precision here! with float the back-prop is very unstable
    a = x[..., 0, 0].double()
    c = x[..., 0, 1].double()
    b = x[..., 1, 1].double()                                # (..., )
    delta = torch.sqrt(a*a + 4*c*c - 2*a*b + b*b)
    v1 = (a - b - delta) / 2. /c
    v1 = torch.stack([v1, torch.ones_like(v1, dtype=torch.double, device=v1.device)], dim=-1)    # (..., 2)
    v2 = (a - b + delta) / 2. /c
    v2 = torch.stack([v2, torch.ones_like(v2, dtype=torch.double, device=v2.device)], dim=-1)    # (..., 2)
    n1 = torch.sum(v1*v1, keepdim=True, dim=-1).sqrt()
    n2 = torch.sum(v2*v2, keepdim=True, dim=-1).sqrt()
    v1 = v1 / n1
    v2 = v2 / n2
    return v1.float(), v2.float()