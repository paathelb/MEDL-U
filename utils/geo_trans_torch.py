import math

import torch
import torch.nn.functional as F

class GeoTransTorch(object):
    """
    GeoTransTorch use torch to do 3D transforms, N objects is process in every method.
    """
    @staticmethod
    def cart2hom(pts_3d):
        """
        Convert Cartesian point to Homogeneous
        :param pts_3d: [N, K, 3] points in Cartesian
        :return: [N, K, 3] points in Homogeneous by pending 1
        """
        n, k, _ = pts_3d.shape
        pts_3d_hom = torch.cat((pts_3d, torch.ones((n, k, 1), device=pts_3d.device)), dim=-1)
        return pts_3d_hom

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    @staticmethod
    def project_rect_to_image(pts_3d_rect, P):
        """

        :param pts_3d_rect: [N, p c, 3] points in rect camera coord.
        :param P: projection matrix, shape = [N, 3, 4]
        :return: [N, K, 2]
        """
        pts_3d_rect = GeoTransTorch.cart2hom(pts_3d_rect)
        pts_2d = torch.matmul(pts_3d_rect, P.permute(0, 2, 1))  # [N, K, 3]

        # use clone to avoid in-placed operation
        pts_2d_clone = pts_2d.clone()
        pts_2d_clone[:, :, 0] /= pts_2d[:, :, 2]
        pts_2d_clone[:, :, 1] /= pts_2d[:, :, 2]
        return pts_2d_clone[:, :, 0:2]

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    @staticmethod
    def project_image_to_rect(uv, depth, P):
        """
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        :param uv: projection points in image, shape = [N, 2]
        :param depth: depth in rect camera coord, shape = [N, ]
        :param P: projection matrix, shape = [N, 3, 4]
        :return: nx3 points in rect camera coord. [N, 3]
        """
        z = depth + P[:, 2, 3]
        x = (uv[:, 0] * z - P[:, 0, 3] - P[:, 0, 2] * depth) / P[:, 0, 0]
        y = (uv[:, 1] * z - P[:, 1, 3] - P[:, 1, 2] * depth) / P[:, 1, 1]

        pts_3d_rect = torch.stack([x.float(), y.float(), depth.float()], dim=1)
        return pts_3d_rect

    # ===========================
    # ------- others ------------
    # ===========================
    @staticmethod
    def rot_mat_y(rot_y):
        """

        :param rot_y: [N, ]
        :return: [N, 3, 3]
        """
        device = rot_y.device
        N = rot_y.shape[0]
        cos, sin = rot_y.cos(), rot_y.sin()
        i_temp = torch.tensor([[1, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 1]]).to(dtype=rot_y.dtype,
                                               device=device)
        ry = i_temp.repeat(N, 1).view(N, -1, 3)

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

    @staticmethod
    def encode_box3d(rotys, dims, locs):
        """
        construct 3d bounding box for each object.
        Args:
            rotys: rotation in shape N
            dims: dimensions of objects, (l, h, w), shape = [N, 3]
            locs: locations of objects, (x, y, z)

        Returns:
            3D bbox: shape of [N, 8, 3]
        """
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = dims.view(-1, 3)
        if len(locs.shape) == 3:
            locs = locs.view(-1, 3)

        device = rotys.device
        N = rotys.shape[0]
        ry = GeoTransTorch.rot_mat_y(rotys)

        dims = dims.reshape(-1, 1).repeat(1, 8)
        dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
        dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
        dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
        index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                              [4, 5, 0, 1, 6, 7, 2, 3],
                              [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
        box_3d_object = torch.gather(dims, 1, index)
        box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
        box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

        return box_3d.permute(0, 2, 1).contiguous()

    @staticmethod
    def encode_box2d(rotys, dims, locs, K, img_size, bound_corners=True):
        """
        Only support objects in a single image, because of img_size
        :param K:
        :param rotys:
        :param dims:
        :param locs:
        :param img_size: [w, h]
        :return: bboxfrom3d, shape = [N, 4]
        """
        device = rotys.device
        K = K.to(device=device)

        box3d = GeoTransTorch.encode_box3d(rotys, dims, locs)
        box3d_image = GeoTransTorch.project_rect_to_image(box3d, K)

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

        return bboxfrom3d


    @staticmethod
    def box3d_to_2d(box3d, K, img_size):
        """
         Only support objects in a single image, because of img_size
        :param K:
        :bboxfrom3d
        :param img_size: [w, h]
        :return: bboxfrom2d, shape = [N, 4]
        """
        box3d_image = GeoTransTorch.project_rect_to_image(box3d, K)

        xmins, _ = box3d_image[:, :, 0].min(dim=1)
        xmaxs, _ = box3d_image[:, :, 0].max(dim=1)
        ymins, _ = box3d_image[:, :, 1].min(dim=1)
        ymaxs, _ = box3d_image[:, :, 1].max(dim=1)


        xmins = xmins.clamp(0, img_size[0])
        xmaxs = xmaxs.clamp(0, img_size[0])
        ymins = ymins.clamp(0, img_size[1])
        ymaxs = ymaxs.clamp(0, img_size[1])

        boxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                               xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)


        return boxfrom3d


    @staticmethod
    def ry_to_alpha(location, ry):
        # TODO a problem: ry_local and alpha in anno always has s samll difference
        # [-pi, pi]
        try:
            ray = torch.atan2(location[:, 2], location[:, 0])
        except:
            location = location.reshape(-1,3)
            ray = torch.atan2(location[:, 2], location[:, 0])
        alpha = ry - (-ray)
        alpha = alpha - 0.5 * math.pi
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        return alpha

    @staticmethod
    def alpha_to_ry(location, alpha):
        # [-pi, pi]
        ray = torch.atan2(location[:, 2], location[:, 0])
        ry = alpha + (-ray)

        ry = ry + 0.5 * math.pi
        ry = (ry + math.pi) % (2 * math.pi) - math.pi
        return ry