from lib.utils.once_utils import generate_dataparser_outputs
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_PCA_Norm, get_Sphere_Norm
from lib.config import cfg
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import cv2
import sys
import copy
import shutil
sys.path.append(os.getcwd())

def readOnceInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):
    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)
        
        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        save_dir = os.path.join(cfg.model_path, 'colmap')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(colmap_dir, save_dir)
        
    output = generate_dataparser_outputs(datadir=path)

    cam_infos = output['cam_infos']
    num_frames = output['num_frames']

    scene_metadata = dict()
    scene_metadata['num_frames'] = num_frames
    
    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]
        
    novel_view_cam_infos = []
    
    #######################################################################################################################3
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    
    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    
    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata    
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    # Get sphere center
    ply_path = output['ply_path']
    bkgd_ply_path = ply_path['bkgd_ply_path']
    lidar_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_lidar.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd: BasicPointCloud = fetchPly(lidar_ply_path)
    else:
        sphere_pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    
    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']
    print(f'Sphere extent: {sphere_normalization["radius"]}')

    point_cloud = output['point_cloud']

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )
    
    return scene_info
    