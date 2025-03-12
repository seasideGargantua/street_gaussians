import torch
from mixplat.projection import project_gaussians
from mixplat.rasterization import rasterize_gaussians
from lib.utils.sh_utils import eval_sh
from lib.models.mix_gaussian_model import MixGaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer, camera_cv2gl
from lib.config import cfg

class MixGaussianRenderer():
    def __init__(
        self,         
    ):
        self.cfg = cfg.render
              
    def render_all(
        self, 
        viewpoint_camera: Camera,
        pc: MixGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        
        # render all
        render_composition = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # render background
        render_background = self.render_background(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        # render object
        render_dynamic = self.render_dynamic(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_dynamic['rgb']
        result['acc_object'] = render_dynamic['acc']
        
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
    
        return result
    
    def render_dynamic(
        self, 
        viewpoint_camera: Camera,
        pc: MixGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        parse_camera_again: bool = True,
    ):        
        pc.set_visibility(include_list=['dynamic'])
        if parse_camera_again: pc.parse_camera(viewpoint_camera)        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_background(
        self, 
        viewpoint_camera: Camera,
        pc: MixGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        parse_camera_again: bool = True,
    ):
        include_list=['background', 'sky']
        exclude_list = list(set(pc.model_name_id.keys()) - set(include_list))
        result = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, exclude_list=exclude_list)
        return result
    
    def render_sky(
        self, 
        viewpoint_camera: Camera,
        pc: MixGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        parse_camera_again: bool = True,
    ):  
        include_list=['sky']
        exclude_list = list(set(pc.model_name_id.keys()) - set(include_list))
        result = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, exclude_list=exclude_list)
        return result
    
    def render(
        self, 
        viewpoint_camera: Camera,
        pc: MixGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
    ):
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))

        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # Step2: render sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)

        return result
    
            
    def render_kernel(
        self, 
        viewpoint_camera: Camera,
        pc: MixGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        white_background = cfg.data.white_background,
    ):
        try:
            means3D = pc.get_xyz
            num_gaussians = len(means3D)
        except:
            num_gaussians = 0
        
        if num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        viewmat, T = camera_cv2gl(viewpoint_camera.R, viewpoint_camera.T)

        timestamp = viewpoint_camera.meta['timestamp']
        cov3ds, xyzs, rgbs, opacity = pc.process_render(timestamp, T)

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        
        # project 3D Gaussians to 2D screen space
        xys, depths, radii, conics, _, num_tiles_hit = project_gaussians(  # type: ignore
            xyzs,
            cov3ds,
            viewmat[:3, :],
            viewpoint_camera.fx,
            viewpoint_camera.fy,
            viewpoint_camera.cx,
            viewpoint_camera.cy,
            viewpoint_camera.image_height,
            viewpoint_camera.image_width,
            16,
        )

        try:
            xys.retain_grad()
        except:
            pass

        rendered_color, rendered_acc, invdepth = rasterize_gaussians(  # type: ignore
                    xys,
                    depths,
                    radii,
                    conics,
                    num_tiles_hit,
                    rgbs,
                    opacity,
                    viewpoint_camera.image_height,
                    viewpoint_camera.image_width,
                    16,
                    background=bg_color,
                    return_alpha=True,
                    return_invdepth=True,
                )
        rendered_color = rendered_color.permute(2, 0, 1)
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": invdepth.unsqueeze(0),
            "viewspace_points": xys,
            "visibility_filter" : radii > 0,
            "radii": radii
        }
        
        return result