import torch
import torch.nn as nn
import numpy as np
import os
from mixplat.projection import compute_4d_gaussians_covariance
from mixplat.sh import spherical_harmonics_3d_fast
from lib.config import cfg
from lib.models.gaussian_model import GaussianModel
from lib.utils.general_utils import quaternion_to_matrix, inverse_sigmoid, matrix_to_quaternion, get_expon_lr_func, quaternion_raw_multiply, build_rotation_4d, build_scaling_rotation_4d
from lib.utils.sh_utils import RGB2SH, IDFT
from lib.datasets.base_readers import fetchPly
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

class GaussianModelDynamic(GaussianModel):
    def __init__(
        self, 
        model_name,
        frame_nums,
        time_duration_start=0.0,
        time_duration_end=1.0,
        init_scale_f=15,
        perframe_sparse_scale_t_mult=1.0,
    ):  
        self.frame_nums = frame_nums
        # fourier spherical harmonics
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)
        self.fourier_scale = cfg.model.gaussian.get('fourier_scale', 1.)
        
        extent = cfg.get('dynamic_extent', 1.0)
        self.extent = torch.tensor([extent]).float().cuda()   

        num_classes = 1 if cfg.data.get('use_semantic', False) else 0
        self.num_classes_global = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0        
        super().__init__(model_name=model_name, num_classes=num_classes)
    
        self.spatial_lr_scale = extent

        self.cov3ds = torch.empty(0)
        self.cov_t = torch.empty(0)
        self.speed = torch.empty(0)

        self._scaling_t = torch.empty(0)
        self._rotation_r = torch.empty(0)
        self._t = torch.empty(0)
        self.time_duration_start = time_duration_start
        self.time_duration_end = time_duration_end
        self.perframe_sparse_scale_t_mult = perframe_sparse_scale_t_mult
        self.init_scale_f = init_scale_f
    
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)

    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)

    def get_delta_xyz(self, ts):
        dt = ts - self._t
        self.delta_xyz = self.speed * dt
        return self.delta_xyz

    @property
    def get_xyz(self):
        if hasattr(self, 'delta_xyz'):
            xyz = self._xyz + self.delta_xyz
        else:
            xyz = self._xyz
        return xyz

    @property
    def get_scaling_xyzt(self):
        return torch.exp(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)

    @property
    def get_gaussians_num(self):
        return self._xyz.shape[0]

    @property
    def get_cov3ds(self):
        self.cov3ds, self.cov_t, self.speed = compute_4d_gaussians_covariance(self.get_scaling, self.get_scaling_t, self.get_rotation, self.get_rotation_r)
        return self.cov3ds

    def get_opacity(self, ts=None):
        if ts is not None:
            dt = ts - self._t
            tshift = 0.5 * dt * dt / self.cov_t
            opacity = self.opacity_activation(self._opacity) * torch.exp(-tshift)
        else:
            opacity = self.opacity_activation(self._opacity)
        return opacity

    def get_rgbs(self, translation):
        if self.active_sh_degree > 0:
            n = self.active_sh_degree
            viewdirs = self.get_xyz.detach() - translation  # (N, 1, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics_3d_fast(n, viewdirs, self.get_features)
        else:
            rgbs = torch.sigmoid(self.get_features[:,0,:])
        return rgbs

    def get_features_fourier(self, frame=0):
        normalized_frame = (frame - self.start_frame) / (self.end_frame - self.start_frame)
        time = self.fourier_scale * normalized_frame

        idft_base = IDFT(time, self.fourier_dim)[0].cuda()
        features_dc = self._features_dc # [N, C, 3]
        features_dc = torch.sum(features_dc * idft_base[..., None], dim=1, keepdim=True) # [N, 1, 3]
        features_rest = self._features_rest # [N, sh, 3]
        features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
        return features
           
    def create_from_pcd(self, pcd, spatial_lr_scale):
        pointcloud_xyz = np.asarray(pcd.points)
        pointcloud_rgb = np.asarray(pcd.colors)
        pointcloud_time = np.asarray(pcd.timestamps)
        
        fused_point_cloud = torch.tensor(np.asarray(pointcloud_xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pointcloud_rgb)).float().cuda())
        fused_times = torch.tensor(pointcloud_time).unsqueeze(1).float().cuda()

        features_dc = torch.zeros((fused_color.shape[0], 3, self.fourier_dim)).float().cuda()
        features_rest = torch.zeros(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1).float().cuda()
        features_dc[:, :3, 0] = fused_color

        print(f"Number of points at initialisation for {self.model_name}: ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pointcloud_xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        duration = self.time_duration_end - self.time_duration_start
        frame_time = duration / self.frame_nums
        init_scale_t = ((frame_time * self.init_scale_f * self.perframe_sparse_scale_t_mult) ** 2 / (np.log(0.05) / -0.5)) ** 2
        dist_t = torch.zeros_like(fused_times, device="cuda") + init_scale_t
        scales_t = torch.log(torch.sqrt(dist_t))

        rots = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
        rots[:, 0] = 1
        rots_r = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
        rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1))).float().cuda()
        semantics = torch.zeros((fused_point_cloud.shape[0], self.num_classes)).float().cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic = nn.Parameter(semantics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self):
        args = cfg.optim

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.t_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = 0
                
        tag = 'dynamic'
        position_lr_init = args.get('position_lr_init_{}'.format(tag), args.position_lr_init)
        position_lr_final = args.get('position_lr_final_{}'.format(tag), args.position_lr_final)
        t_lr_init = args.get('t_lr_init_{}'.format(tag), args.t_lr_init)
        t_lr_final = args.get('t_lr_final_{}'.format(tag), args.t_lr_final)
        scaling_lr = args.get('scaling_lr_{}'.format(tag), args.scaling_lr)
        scaling_t_lr = args.get('scaling_t_lr_{}'.format(tag), args.scaling_lr)
        feature_lr = args.get('feature_lr_{}'.format(tag), args.feature_lr)
        semantic_lr = args.get('semantic_lr_{}'.format(tag), args.semantic_lr)
        rotation_lr = args.get('rotation_lr_{}'.format(tag), args.rotation_lr)
        rotation_r_lr = args.get('rotation_r_lr_{}'.format(tag), args.rotation_lr)
        opacity_lr = args.get('opacity_lr_{}'.format(tag), args.opacity_lr)
        feature_rest_lr = args.get('feature_rest_lr_{}'.format(tag), feature_lr / 20.0)

        l = [
            {'params': [self._xyz], 'lr': position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_rest_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': semantic_lr, "name": "semantic"},
            {'params': [self._rotation_r], 'lr': rotation_r_lr, "name": "rotation_r"},
            {'params': [self._t], 'lr': t_lr_init, "name": "t"},
            {'params': [self._scaling_t], 'lr': scaling_t_lr, "name": "scaling_t"},
        ]
        
        self.percent_dense = args.percent_dense
        self.percent_dense_t = args.percent_dense_t
        self.percent_big_ws = args.percent_big_ws
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init * self.spatial_lr_scale,
            lr_final=position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps
        )
        self.t_scheduler_args = get_expon_lr_func(
            lr_init=t_lr_init,
            lr_final=t_lr_final,
            lr_delay_mult=0.01,
            max_steps=30000
        )

        self.densify_and_prune_list = ['xyz, f_dc, f_rest, opacity, scaling, rotation, semantic']
        self.scalar_dict = dict()
        self.tensor_dict = dict()  
    
    def state_dict(self, is_final=False):
        state_dict = super().state_dict(is_final=is_final)
        state_dict_nonrigid = {
            'scaling_t': self._scaling_t,
            'rotation_r': self._rotation_r,
            't': self._t,
        }
        state_dict.update(state_dict_nonrigid)
        
        return state_dict

    def load_state_dict(self, state_dict):  
        self._xyz = state_dict['xyz']  
        self._t = state_dict['t']
        self._features_dc = state_dict['feature_dc']
        self._features_rest = state_dict['feature_rest']
        self._scaling = state_dict['scaling']
        self._scaling_t = state_dict['scaling_t']
        self._rotation = state_dict['rotation']
        self._rotation_r = state_dict['rotation_r']
        self._opacity = state_dict['opacity']
        self._semantic = state_dict['semantic']
        
        if cfg.mode == 'train':
            self.training_setup()
            if 'spatial_lr_scale' in state_dict:
                self.spatial_lr_scale = state_dict['spatial_lr_scale'] 
            if 'denom' in state_dict:
                self.denom = state_dict['denom'] 
            if 'max_radii2D' in state_dict:
                self.max_radii2D = state_dict['max_radii2D'] 
            if 'xyz_gradient_accum' in state_dict:
                self.xyz_gradient_accum = state_dict['xyz_gradient_accum']
            if 't_gradient_accum' in state_dict:
                self.t_gradient_accum = state_dict['t_gradient_accum']
            if 'active_sh_degree' in state_dict:
                self.active_sh_degree = state_dict['active_sh_degree']
            if 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])

    def densify_and_prune(self, ts, max_grad, max_grad_t, min_opacity, prune_big_points):
        max_grad = cfg.optim.get('densify_grad_threshold_dynamic', max_grad)
        if cfg.optim.get('densify_grad_abs_dynamic', False):
            grads = self.xyz_gradient_accum[:, 1:2] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        
        grads[grads.isnan()] = 0.0

        grads_t = self.t_gradient_accum / self.denom
        grads_t[grads_t.isnan()] = 0.0
        grads_t = grads_t.squeeze()

        # Clone and Split
        # extent = self.get_extent()
        extent = self.extent
        self.densify_and_clone(grads, grads_t, max_grad, max_grad_t, extent)
        self.densify_and_split(grads, grads_t, max_grad, max_grad_t, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity(ts) < min_opacity).squeeze()
        # Prune points with big scale
        # big_points_ws = self.get_scaling.max(dim=1).values > extent * self.percent_big_ws

        # prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)
        
        # Reset
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        
        return self.scalar_dict, self.tensor_dict
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        prune_list = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'semantic', 't', 'scaling_t', 'rotation_r']
        optimizable_tensors = self.prune_optimizer(valid_points_mask, prune_list = prune_list)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]

        # 4dgs
        self._t = optimizable_tensors["t"]
        self._scaling_t = optimizable_tensors["scaling_t"]
        self._rotation_r = optimizable_tensors["rotation_r"]
        self.cov_t = self.cov_t[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def densify_and_split(self, grads, grads_t, grad_threshold, grad_threshold_t, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
                
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_extent = torch.zeros((n_init_points), device="cuda")
        padded_extent[:grads.shape[0]] = scene_extent

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * padded_extent)
        
        padded_grad_t = torch.zeros((n_init_points), device="cuda")
        padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()

        selected_pts_mask_t = torch.where(padded_grad_t >= grad_threshold_t, True, False)
        selected_pts_mask_t = torch.logical_and(selected_pts_mask_t,
            self.get_scaling_t.squeeze() > self.percent_dense_t)

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_t)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 4), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
        new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
        new_xyz = new_xyzt[...,0:3]
        new_t = new_xyzt[...,3:4]
        new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N, 1)
        new_semantic = self._semantic[selected_pts_mask].repeat(N, 1)

        new_cov_t = self.cov_t[selected_pts_mask].repeat(N, 1)
        self.cov_t = torch.cat([self.cov_t, new_cov_t], dim=0)

        densification_dict = {
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacity, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "t" : new_t,
            "scaling_t" : new_scaling_t,
            "rotation_r" :new_rotation_r,
            "semantic" : new_semantic,
        }

        self.densification_postfix(densification_dict)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grads_t, grad_threshold, grad_threshold_t, scene_extent):
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        selected_pts_mask_t = torch.where(grads_t >= grad_threshold_t, True, False)
        selected_pts_mask_t = torch.logical_and(selected_pts_mask_t,
            self.get_scaling_t.squeeze() <= self.percent_dense_t)

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_t)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_t = self._t[selected_pts_mask]
        new_scaling_t = self._scaling_t[selected_pts_mask]
        new_rotation_r = self._rotation_r[selected_pts_mask]
        new_semantic = self._semantic[selected_pts_mask]

        new_cov_t = self.cov_t[selected_pts_mask]
        self.cov_t = torch.cat([self.cov_t, new_cov_t], dim=0)

        densification_dict = {
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacity, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "t": new_t,
            "cov_t": new_cov_t,
            "scaling_t": new_scaling_t,
            "rotation_r": new_rotation_r,
            "semantic" : new_semantic,
        }

        self.densification_postfix(densification_dict)

    def densification_postfix(self, tensors_dict, reset_params=True):
        optimizable_tensors = self.cat_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]
        
        self._t = optimizable_tensors["t"]
        self._scaling_t = optimizable_tensors["scaling_t"]
        self._rotation_r = optimizable_tensors["rotation_r"]

        if reset_params:
            cat_points_num = self.get_xyz.shape[0] - self.xyz_gradient_accum.shape[0]
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(cat_points_num, 2).cuda()], dim=0)
            self.t_gradient_accum = torch.cat([self.t_gradient_accum, torch.zeros(cat_points_num, 1).cuda()], dim=0)
            self.denom = torch.cat([self.denom, torch.zeros(cat_points_num, 1).cuda()], dim=0)
            self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(cat_points_num).cuda()], dim=0)

    def set_max_radii(self, visibility_dynamic, max_radii2D):
        self.max_radii2D[visibility_dynamic] = torch.max(self.max_radii2D[visibility_dynamic], max_radii2D[visibility_dynamic])
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity(), torch.ones_like(self.get_opacity()) * 0.01))
        d = {'opacity': opacities_new}
        optimizable_tensors = self.reset_optimizer(d)
        self._opacity = optimizable_tensors["opacity"]