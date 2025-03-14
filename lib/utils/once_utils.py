import os
import numpy as np
import cv2
import open3d as o3d
import torch
import math
from glob import glob
from tqdm import tqdm
from lib.config import cfg
from lib.datasets.base_readers import storePly, fetchPly, get_Sphere_Norm
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames, uniform_sample_sphere, check_pts_visibility
from lib.utils.colmap_utils import read_points3D_binary
from lib.datasets.base_readers import CameraInfo
from lib.utils.once_devkit import ONCE
from lib.utils.img_utils import visualize_depth_numpy

_cam_name_id_map = {
    'cam01': 0,
    'cam03': 1,
    'cam05': 2,
    'cam06': 3,
    'cam07': 4,
    'cam08': 5,
    'cam09': 6,}

_cam_id_name_map = {
    0: 'cam01',
    1: 'cam03',
    2: 'cam05',
    3: 'cam06',
    4: 'cam07',
    5: 'cam08',
    6: 'cam09',}

def generate_dataparser_outputs(datadir):
    seq_id = cfg.data.seq_id
    cam_names = cfg.data.get('cam_names', ['cam03']) # default to ['cam03']
    selected_frames = cfg.data.get('selected_frames', [0, -1])
    once_loader = ONCE(datadir, seq_id)
    frame_list = once_loader.get_frame_ids(cam_names[0])[selected_frames[0]:selected_frames[1]]
    num_frames = len(frame_list)

    # load camera, frame, path
    cams = []
    image_filenames = []
    frames_idx = []

    ixts = []
    c2ws = []

    points_list = []

    cam_infos = []

    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )
    
    depth_dir = os.path.join(cfg.model_path, 'lidar_depth')
    build_lidar_depth = (not os.path.exists(depth_dir) or cfg.data.get('build_lidar_depth', False)) or (len(os.listdir(depth_dir)) == 0)
    if build_lidar_depth:
        os.makedirs(depth_dir, exist_ok=True)

    ply_dir = os.path.join(cfg.model_path, 'input_ply')
    bkgd_ply_path = os.path.join(ply_dir, 'points3D_bkgd.ply')
    dynamic_ply_path = os.path.join(ply_dir, 'points3D_dynamic.ply')
    build_pointcloud = (cfg.mode == 'train') and (not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False))
    if build_pointcloud:
        os.makedirs(ply_dir, exist_ok=True)

    W,H = once_loader.get_WH()
    poss = []
    for idx, frame_id in tqdm(enumerate(frame_list)):
        pos = once_loader.get_l2w(frame_id)[:3, 3]
        poss.append(pos)
    offset = np.array(poss).mean()
    for idx, frame_id in tqdm(enumerate(frame_list), desc="Loading data"):
        timestamp = idx / num_frames
        if build_pointcloud:
            points_xyz = once_loader.load_point_cloud(frame_id)[:, :3]
            points_time = np.full_like(points_xyz[:, :1], timestamp)
        for cam_name in cam_names:
            cam_id = _cam_name_id_map[cam_name]

            image = once_loader.load_image(frame_id, cam_name)
            image_path = once_loader.get_image_path(frame_id, cam_name)
            image_name = os.path.basename(image_path)
            obj_bound = once_loader.load_obj_bound(frame_id, cam_name)

            ixt = once_loader.get_intr(cam_name)
            c2w = once_loader.get_c2w(frame_id, cam_name, offset=offset)
            l2w = once_loader.get_l2w(frame_id)

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = np.array(w2c[:3,3])

            cams.append(cam_id)
            image_filenames.append(image_path)
            frames_idx.append(idx)

            ixts.append(ixt)
            c2ws.append(c2w)

            fx, fy = ixt[0, 0], ixt[1, 1]
            cx, cy = ixt[0, 2], ixt[1, 2]
            FovY = focal2fov(fx, H)
            FovX = focal2fov(fy, W)

            if build_pointcloud:
                points_xyz_homo = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
                points_xyz_world = (points_xyz_homo @ l2w.T)[:, :3] - offset
                points_dict = once_loader.split_point_cloud(points_xyz_world, points_time, image, obj_bound, w2c, ixt, W, H)
                points_list.append(points_dict)

            if build_lidar_depth:
                depth = once_loader.load_lidar_depth(frame_id, cam_name)
                depth_vis, _ = visualize_depth_numpy(depth)
                depth_path = os.path.join(depth_dir, f'{frame_id}.npy')
                np.save(depth_path, depth)
                depth_vis_path = os.path.join(depth_dir, f'{frame_id}.png')
                cv2.imwrite(depth_vis_path, depth_vis)
            else:
                depth_path = os.path.join(depth_dir, f'{frame_id}.npy')
                depth = np.load(depth_path).astype(np.float32)

            metadata = dict()
            metadata['frame'] = frame_id
            metadata['cam'] = cam_id
            metadata['frame_idx'] = idx
            metadata['timestamp'] = timestamp

            if idx in train_frames:
                metadata['is_val'] = False
            else:
                metadata['is_val'] = True

            guidance = dict()
            guidance['sky_mask'] = once_loader.load_sky_mask(frame_id, cam_name)
            guidance['obj_bound'] = obj_bound
            guidance['lidar_depth'] = depth

            cam_info = CameraInfo(
                uid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX, K=ixt,
                image=image, image_path=image_path, image_name=image_name,
                width=W, height=H,
                metadata=metadata,
                guidance=guidance,
            )
            cam_infos.append(cam_info)

            
    ixts = np.stack(ixts, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    result = dict()
    result['ixts'] = ixts
    result['c2ws'] = c2ws
    result['cams'] = cams
    result['image_filenames'] = image_filenames
    result['frames_idx'] = frames_idx
    result['cam_infos'] = cam_infos
    result['ply_path'] = {'bkgd_ply_path': bkgd_ply_path, 'dynamic_ply_path': dynamic_ply_path}
    result['calib'] = once_loader.calib
    result['num_frames'] = num_frames
          
    # run colmap
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from script.once.colmap_once import run_colmap_once
        run_colmap_once(result)
    
    if build_pointcloud:
        print('build point cloud')
        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_time_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        points_xyz_dict['dynamic'] = []
        points_rgb_dict['dynamic'] = []
        points_time_dict['dynamic'] = []

        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.
                     
        print('initialize from lidar pointcloud')
        for i, points in tqdm(enumerate(points_list)):
            points_xyz_dict['bkgd'].append(points['bkgd_points'])
            points_rgb_dict['bkgd'].append(points['bkgd_points_rgb'])
            points_xyz_dict['dynamic'].append(points['dynamic_points'])
            points_rgb_dict['dynamic'].append(points['dynamic_points_rgb'])
            points_time_dict['dynamic'].append(points['dynamic_points_time'])
        
        initial_num_dynamic = cfg.data.get('initial_num_dynamic', 100000)
        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    if cfg.data.get("downsample", False):
                        # downsample lidar pointcloud with voxels
                        points_lidar = o3d.geometry.PointCloud()
                        points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                        points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                        downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                        downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                        points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                        points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                    else:
                        points_lidar_xyz = points_xyz
                        points_lidar_rgb = points_rgb
                elif k == 'dynamic':
                    points_time = np.concatenate(points_time_dict[k], axis=0)
                    if len(points_xyz) > initial_num_dynamic:
                        random_indices = np.random.choice(len(points_xyz), initial_num_dynamic, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]
                        points_time = points_time[random_indices]
                        
                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb
                    points_time_dict[k] = points_time
                
                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        # combine SfM pointcloud with LiDAR pointcloud
        try:
            if cfg.data.filter_colmap:
                points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
                for i, ext in enumerate(exts):
                    camera_position = c2ws[i][:3, 3]
                    radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                    mask = np.logical_or(radius < cfg.data.get('extent', 10), points_colmap_xyz[:, 2] < camera_position[2])
                    points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)        
                points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
                points_colmap_rgb = points_colmap_rgb[points_colmap_mask]
            
            points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
            mask = points_colmap_dist < 2 * sphere_radius
            points_colmap_xyz = points_colmap_xyz[mask]
            points_colmap_rgb = points_colmap_rgb[mask]
        
            points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0) 
            points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
        except:
            print('No colmap pointcloud')
            points_bkgd_xyz = points_lidar_xyz
            points_bkgd_rgb = points_lidar_rgb
        
        num_far_pts = 300000
        far_rand_pts = uniform_sample_sphere(num_far_pts, 'cpu', inverse=True)
        valid_mask = check_pts_visibility(ixts[0], c2ws, W, H, far_rand_pts)
        far_rand_pts = far_rand_pts[valid_mask].numpy()
        far_rand_rgb = np.random.rand(*far_rand_pts.shape).astype(np.float32)
        print(f"generate far random pts num {far_rand_pts.shape[0]}")
        points_bkgd_xyz = np.concatenate([points_bkgd_xyz, far_rand_pts], axis=0)
        points_bkgd_rgb = np.concatenate([points_bkgd_rgb, far_rand_rgb], axis=0)

        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        points_xyz_dict['colmap'] = points_colmap_xyz
        points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb

        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            if k == 'dynamic':
                points_time = points_time_dict[k]
            else:
                points_time = np.zeros_like(points_xyz[:, 0])
            ply_path = os.path.join(ply_dir, f'points3D_{k}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb, points_time)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue

    point_cloud = dict()
    point_cloud['bkgd'] = fetchPly(bkgd_ply_path)
    point_cloud['dynamic'] = fetchPly(dynamic_ply_path)
    result['point_cloud'] = point_cloud

    return result