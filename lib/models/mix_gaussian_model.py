import torch
import torch.nn as nn
import numpy as np
import os
from simple_knn._C import distCUDA2
from lib.config import cfg
from lib.utils.general_utils import quaternion_to_matrix, \
    build_scaling_rotation, \
    strip_symmetric, \
    quaternion_raw_multiply, \
    startswith_any, \
    matrix_to_quaternion, \
    quaternion_invert
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.models.gaussian_model_dynamic import GaussianModelDynamic
from lib.models.gaussian_model_sky import GaussinaModelSky
from bidict import bidict
from lib.utils.camera_utils import Camera
from lib.utils.sh_utils import eval_sh
from lib.models.sky_cubemap import SkyCubeMap
from lib.models.color_correction import ColorCorrection
from lib.models.camera_pose import PoseCorrection

class MixGaussianModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
            
        self.max_sh_degree = cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        # background + dynamic
        self.include_background = cfg.model.nsg.get('include_bkgd', True)
        self.include_dynamic = cfg.model.nsg.get('include_dynamic', True)
        
        # sky (modeling sky with gaussians, if set to false represent the sky with cube map)
        self.include_sky = cfg.model.nsg.get('include_sky', False) 
        if self.include_sky:
            assert cfg.data.white_background is False

                
        # fourier sh dimensions
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)
        
        # layer color correction
        self.use_color_correction = cfg.model.use_color_correction
        
        # camera pose optimizations (not test)
        self.use_pose_correction = cfg.model.use_pose_correction
    
        # symmetry
        self.flip_prob = cfg.model.gaussian.get('flip_prob', 0.)
        self.flip_axis = 1 
        self.flip_matrix = torch.eye(3).float().cuda() * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))
        self.setup_functions() 
    
    def set_visibility(self, include_list):
        self.include_list = include_list # prefix

    def get_visibility(self, model_name):
        if model_name == 'background':
            if model_name in self.include_list and self.include_background:
                return True
            else:
                return False
        elif model_name == 'sky':
            if model_name in self.include_list and self.include_sky:
                return True
            else:
                return False
        elif model_name == 'dynamic':
            if model_name in self.include_list and self.include_dynamic:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')
                
    def create_from_pcd(self, pcd: dict, spatial_lr_scale: float):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.create_from_pcd(pcd['bkgd'], spatial_lr_scale)
            else:
                model.create_from_pcd(pcd['dynamic'], spatial_lr_scale)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        
        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModel = getattr(self, model_name)
            plydata = model.make_ply()
            plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            plydata_list.append(plydata)

        PlyData(plydata_list).write(path)
        
    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:] # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: GaussianModel = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements
                
        self.active_sh_degree = self.max_sh_degree
  
    def load_state_dict(self, state_dict, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])
            
        if self.sky_cubemap is not None:
            self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])
            
        if self.color_correction is not None:
            self.color_correction.load_state_dict(state_dict['color_correction'])
            
        if self.pose_correction is not None:
            self.pose_correction.load_state_dict(state_dict['pose_correction'])
                            
    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
        
        if self.sky_cubemap is not None:
            state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(is_final)

        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)
    
        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)
    
        return state_dict
        
    def setup_functions(self):
        self.model_name_id = bidict()
        self.models_num = 0
        
        # Build background model
        if self.include_background:
            self.background = GaussianModelBkgd(
                model_name='background', 
                scene_center=self.metadata['scene_center'],
                scene_radius=self.metadata['scene_radius'],
                sphere_center=self.metadata['sphere_center'],
                sphere_radius=self.metadata['sphere_radius'],
            )
                                    
            self.model_name_id['background'] = 0
            self.models_num += 1

        # Build dynamic models
        if self.include_dynamic:
            self.dynamic = GaussianModelDynamic(
                model_name='dynamic',
                frame_nums=self.metadata['num_frames']
            )

            self.model_name_id['dynamic'] = 1
            self.models_num += 1

        # Build sky model
        if self.include_sky:
            self.sky_cubemap = SkyCubeMap()    
        else:
            self.sky_cubemap = None    
                             
        # Build color correction
        if self.use_color_correction:
            self.color_correction = ColorCorrection(self.metadata)
        else:
            self.color_correction = None
            
        # Build pose correction
        if self.use_pose_correction:
            self.pose_correction = PoseCorrection(self.metadata)
        else:
            self.pose_correction = None
            
        
    def parse_camera(self, camera: Camera):
        # set camera
        self.viewpoint_camera = camera
        
        # set background mask
        self.background.set_background_mask(camera)
        
        self.frame = camera.meta['frame']
        self.frame_idx = camera.meta['frame_idx']
        self.frame_is_val = camera.meta['is_val']
        self.num_gaussians = 0
        self.graph_gaussian_range = dict()
        idx = 0

        # background        
        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.num_gaussians += num_gaussians_bkgd
            self.graph_gaussian_range['background'] = [idx, idx + num_gaussians_bkgd]
            idx += num_gaussians_bkgd
        
        # dynamic
        if self.get_visibility('dynamic'):
            num_gaussians_dynamic = self.dynamic._xyz.shape[0]
            self.num_gaussians += num_gaussians_dynamic
            self.graph_gaussian_range['dynamic'] = [idx, idx + num_gaussians_dynamic]
            idx += num_gaussians_dynamic
    
    def get_xyz(self, ts=None):
        xyzs = []
        if self.get_visibility('background'):
            xyz_bkgd = self.background.get_xyz
            if self.use_pose_correction:
                xyz_bkgd = self.pose_correction.correct_gaussian_xyz(self.viewpoint_camera, xyz_bkgd)
            
            xyzs.append(xyz_bkgd)
        
        if self.get_visibility('dynamic'):
            xyz_dynamic = self.dynamic.get_xyz(ts)
            if self.use_pose_correction:
                xyz_dynamic = self.pose_correction.correct_gaussian_xyz(self.viewpoint_camera, xyz_dynamic)
            
            xyzs.append(xyz_dynamic)

        xyzs = torch.cat(xyzs, dim=0)

        return xyzs            
    
    def get_colors(self, camera_center):
        colors = []
        if self.get_visibility('background'):
            color_bkgd = self.background.get_rgbs(camera_center)
            colors.append(color_bkgd)
        
        if self.get_visibility('dynamic'):
            color_dynamic = self.dynamic.get_rgbs(camera_center)
            colors.append(color_dynamic)

        colors = torch.cat(colors, dim=0)
        return colors
                   
    def get_opacity(self, ts):
        opacities = []
        
        if self.get_visibility('background'):
            opacity_bkgd = self.background.get_opacity
            opacities.append(opacity_bkgd)

        if self.get_visibility('dynamic'):
            opacity_dynamic = self.dynamic.get_opacity(ts)
            # opacity_dynamic = self.dynamic.get_opacity()
            opacities.append(opacity_dynamic)
        
        opacities = torch.cat(opacities, dim=0)
        return opacities
    
    @property
    def get_cov3ds(self):
        cov3ds = []
        if self.get_visibility('background'):
            cov3ds_bkgd = self.background.get_cov3ds
            cov3ds.append(cov3ds_bkgd)
        
        if self.get_visibility('dynamic'):
            cov3ds_dynamic = self.dynamic.get_cov3ds
            cov3ds.append(cov3ds_dynamic)

        cov3ds = torch.cat(cov3ds, dim=0)
        return cov3ds
    
    @property
    def get_gaussians_num(self):
        num = 0
        if self.get_visibility('background'):
            num += self.background._xyz.shape[0]

        if self.get_visibility('dynamic'):
            num += self.dynamic._xyz.shape[0]

        return num

    def process_render(self, ts, camera_center):
        cov3ds = self.get_cov3ds
        opacity = self.get_opacity(ts)
        xyzs = self.get_xyz(ts)
        rgbs = self.get_colors(camera_center)
        return cov3ds, xyzs, rgbs, opacity

    def get_normals(self, camera: Camera):
        normals = []
        
        if self.get_visibility('background'):
            normals_bkgd = self.background.get_normals(camera)            
            normals.append(normals_bkgd)
            
        normals = torch.cat(normals, dim=0)
        return normals
            
    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            model: GaussianModel = getattr(self, model_name)
            model.oneupSHdegree()
                    
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.training_setup()
                
        if self.sky_cubemap is not None:
            self.sky_cubemap.training_setup()
            
        if self.color_correction is not None:
            self.color_correction.training_setup()
            
        if self.pose_correction is not None:
            self.pose_correction.training_setup()
        
    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_learning_rate(iteration)
        
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_learning_rate(iteration)
            
        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)
            
        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)
    
    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_optimizer()

        if self.sky_cubemap is not None:
            self.sky_cubemap.update_optimizer()
            
        if self.color_correction is not None:
            self.color_correction.update_optimizer()
            
        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def set_max_radii2D(self, radii, visibility_filter, exclude_list=[]):
        radii = radii.float()
        
        for model_name in self.graph_gaussian_range.keys():
            if model_name in exclude_list:
                continue
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            visibility_model = visibility_filter[start:end]
            max_radii2D_model = radii[start:end]
            model.max_radii2D[visibility_model] = torch.max(
                model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])
        
    def add_densification_stats(self, viewspace_point_tensor, visibility_filter, exclude_list=[]):
        viewspace_point_tensor_grad = viewspace_point_tensor.grad.squeeze()
        for model_name in self.graph_gaussian_range.keys():
            if model_name in exclude_list:
                continue
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            visibility_model = visibility_filter[start:end]
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end]
            model.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, 2:], dim=-1, keepdim=True)
            model.denom[visibility_model] += 1
            if model_name == 'dynamic':
                if model._t.grad is None:
                    continue
                model.t_gradient_accum[visibility_model] += model._t.grad.clone()[visibility_model].detach()
        
    def densify_and_prune(self, ts, max_grad, max_grad_t, min_opacity, prune_big_points, exclude_list=[]):
        scalars = None
        tensors = None
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            if model_name == 'dynamic':
                scalars_, tensors_ = model.densify_and_prune(ts, max_grad, max_grad_t, min_opacity, prune_big_points)
            else:
                scalars_, tensors_ = model.densify_and_prune(max_grad, min_opacity, prune_big_points)

            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_
    
        return scalars, tensors
            
    def reset_opacity(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()
