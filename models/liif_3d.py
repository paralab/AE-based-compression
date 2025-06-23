import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


def make_coord_3d(shape, ranges=None, flatten=True):
    """Make 3D coordinates"""
    D, H, W = shape
    if ranges is None:
        ranges = [(-1, 1), (-1, 1), (-1, 1)]
    
    coord_seqs = []
    for i, n in enumerate(shape):
        r0, r1 = ranges[i]
        coord_seqs.append(torch.linspace(r0, r1, n))
    
    coord_z, coord_y, coord_x = torch.meshgrid(*coord_seqs, indexing='ij')
    coord = torch.stack([coord_x, coord_y, coord_z], dim=-1)  # (D, H, W, 3)
    
    if flatten:
        coord = coord.view(-1, 3)  # (D*H*W, 3)
    
    return coord


@register('liif-3d')
class LIIF3D(nn.Module):
    """3D Local Implicit Image Function"""

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            # For now, we're not using 3D feature unfolding, so no multiplication
            # if self.feat_unfold:
            #     imnet_in_dim *= 27  # 3x3x3 = 27 for 3D
            imnet_in_dim += 3  # attach 3D coord
            if self.cell_decode:
                imnet_in_dim += 3  # 3D cell
            # Add 1 more dimension for scale parameter t = S²
            imnet_in_dim += 1
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None, scale=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1).unsqueeze(1), 
                              mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                .permute(0, 2, 1)
            return ret

        # Simplified approach without complex local ensemble for debugging
        if not self.local_ensemble:
            vx_lst, vy_lst, vz_lst, eps_shift = [0], [0], [0], 0
        else:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1] 
            vz_lst = [-1, 1]
            eps_shift = 1e-6

        # Calculate relative positions for 3D grid sampling
        rx = 2 / feat.shape[-3] / 2
        ry = 2 / feat.shape[-2] / 2
        rz = 2 / feat.shape[-1] / 2

        device = feat.device
        feat_coord = make_coord_3d(feat.shape[-3:], flatten=False).to(device) \
            .permute(3, 0, 1, 2) \
            .unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[-3:])

        preds = []
        areas = []
        
        for vx in vx_lst:
            for vy in vy_lst:
                for vz in vz_lst:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_[:, :, 2] += vz * rz + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    # 3D grid sampling - fix the coordinate order for PyTorch
                    # PyTorch expects (x, y, z) but our coord is (x, y, z) already
                    coord_grid = coord_.flip(-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 3)
                    
                    q_feat = F.grid_sample(
                        feat, coord_grid,
                        mode='bilinear', align_corners=False, padding_mode='border'
                    )[:, :, 0, 0, :].permute(0, 2, 1)  # (B, N, C)
                    
                    q_coord = F.grid_sample(
                        feat_coord, coord_grid,
                        mode='bilinear', align_corners=False, padding_mode='border'
                    )[:, :, 0, 0, :].permute(0, 2, 1)  # (B, N, 3)

                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-3]
                    rel_coord[:, :, 1] *= feat.shape[-2]
                    rel_coord[:, :, 2] *= feat.shape[-1]

                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    if self.cell_decode:
                        rel_cell = cell.clone()
                        rel_cell[:, :, 0] *= feat.shape[-3]
                        rel_cell[:, :, 1] *= feat.shape[-2]
                        rel_cell[:, :, 2] *= feat.shape[-1]
                        inp = torch.cat([inp, rel_cell], dim=-1)

                    # Improved scale handling
                    if scale is not None:
                        bs, q = coord.shape[:2]
                        if scale.dim() == 1:  # (3,) for 3D scale
                            # Use the average scale for isotropic scaling
                            avg_scale = scale.mean()
                            scale_squared = (avg_scale ** 2).unsqueeze(0).unsqueeze(0).expand(bs, q, 1)
                        elif scale.dim() == 2:  # (B, 3)
                            avg_scale = scale.mean(dim=1)  # Average across 3D
                            scale_squared = (avg_scale ** 2).unsqueeze(1).unsqueeze(-1).expand(bs, q, 1)
                        else:
                            scale_squared = scale.unsqueeze(-1)
                    else:
                        # Fallback: estimate from cell size
                        cell_volume = torch.abs(cell[:, :, 0] * cell[:, :, 1] * cell[:, :, 2])
                        scale_squared = torch.clamp(8.0 / (cell_volume + 1e-8), min=1.0, max=64.0)
                        scale_squared = scale_squared.unsqueeze(-1)
                    
                    inp = torch.cat([inp, scale_squared], dim=-1)

                    bs, q = coord.shape[:2]
                    pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds.append(pred)

                    # Calculate area for local ensemble weighting
                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1] * rel_coord[:, :, 2])
                    areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble and len(areas) == 8:
            # For 3D, we have 8 corners - reorder for proper ensemble weighting
            for i in range(4):
                t = areas[i]; areas[i] = areas[7-i]; areas[7-i] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell, scale=None):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell, scale) 