import os
import numpy as np
import cv2
import open3d as o3d
import torch
import math
from glob import glob
from tqdm import tqdm
from lib.config import cfg
from lib.datasets.base_readers import storePly
from lib.utils.graphics_utils import focal2fov
from lib.utils.data_utils import get_val_frames
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

def generate_dataparser_outputs(
        datadir,
        build_pointcloud=False,     
        building_depth=True,
    ):
    seq_id = cfg.data.seq_id
    cam_names = cfg.data.get('cam_names', ['cam03']) # default to ['cam03']
    selected_frames = cfg.data.get('selected_frames', [0, -1])
    once_loader = ONCE(cfg.data.source_path, seq_id)
    frame_list = once_loader.get_frame_ids(cam_names[0])[selected_frames[0]:selected_frames[1]]
    num_frames = len(frame_list)

    # load camera, frame, path
    cams = []
    image_filenames = []
    frames_idx = []

    ixts = []
    c2ws = []

    points = []
    points_time = []

    cam_infos = []

    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )
    
    depth_dir = os.path.join(cfg.model_path, 'lidar_depth')
    build_lidar_depth = not os.path.exists(depth_dir) or cfg.data.get('build_lidar_depth', False)
    if build_lidar_depth:
        os.makedirs(depth_dir, exist_ok=True)

    W,H = once_loader.get_WH()
    for idx, frame_id in tqdm(enumerate(frame_list), desc="Loading data"):
        for cam_name in cam_names:
            cam_id = _cam_name_id_map[cam_name]

            image = once_loader.load_image(frame_id, cam_name)
            image_path = once_loader.get_image_path(frame_id, cam_name)
            image_name = os.path.basename(image_path)

            ixt = once_loader.get_intr(cam_name)
            c2w = once_loader.get_c2w(frame_id, cam_name)
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

            points_xyz = once_loader.load_point_cloud(frame_id)[:, :3]
            points_xyz_world = (np.pad(points_xyz, (0, 1), constant_values=1) @ l2w.T)[:, :3]
            points.append(points_xyz_world)
            timestamp = idx / num_frames
            point_time = np.full_like(points_xyz_world[:, :1], timestamp)
            points_time.append(point_time)

            if building_depth:
                depth = once_loader.load_building_depth(frame_id, cam_name)
                depth_vis, _ = visualize_depth_numpy(depth)
                depth_path = os.path.join(depth_dir, f'{frame_id}.npy')
                np.save(depth_path, depth)
                depth_vis_path = os.path.join(depth_dir, f'{frame_id}.png')
                cv2.imwrite(depth_vis_path, depth_vis)
            else:
                depth_path = os.path.join(depth_dir, f'{frame_id}.npy')
                depth = np.load(depth_path).astype(np.float32)

            metadata = dict()
            metadata['frame'] = frames_id
            metadata['cam'] = cam_id
            metadata['frame_idx'] = idx
            metadata['timestamp'] = timestamp

            if idx in train_frames:
                metadata['is_val'] = False
            else:
                metadata['is_val'] = True

            guidance = dict()
            guidance['sky_mask'] = once_loader.load_sky_mask(frame_id, cam_name)
            guidance['obj_bound'] = once_loader.load_obj_bound(frame_id, cam_name)
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
          
    # run colmap
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from script.waymo.colmap_waymo_full import run_colmap_waymo
        run_colmap_waymo(result)
    
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)
        
        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []

        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.
                     
        print('initialize from lidar pointcloud')
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()

        for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
            idxs = list(range(i * num_cameras, (i+1) * num_cameras))
            cams_frame = [cams[idx] for idx in idxs]
            image_filenames_frame = [image_filenames[idx] for idx in idxs]
            
            raw_3d = pts3d_dict[frame]
            raw_2d = pts2d_dict[frame]
            
            # use the first projection camera
            points_camera_all = raw_2d[..., 0]
            points_projw_all = raw_2d[..., 1]
            points_projh_all = raw_2d[..., 2]

            # each point should be observed by at least one camera in camera lists
            mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)
            
            # get filtered LiDAR pointcloud position and color        
            points_xyz_vehicle = raw_3d[mask]

            # transfrom LiDAR pointcloud from vehicle frame to world frame
            ego_pose = ego_frame_poses[frame]
            points_xyz_vehicle = np.concatenate(
                [points_xyz_vehicle, 
                np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
            )
            points_xyz_world = points_xyz_vehicle @ ego_pose.T
            
            points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
            points_camera = points_camera_all[mask]
            points_projw = points_projw_all[mask]
            points_projh = points_projh_all[mask]

            for cam, image_filename in zip(cams_frame, image_filenames_frame):
                mask_cam = (points_camera == cam)
                image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.

                mask_projw = points_projw[mask_cam]
                mask_projh = points_projh[mask_cam]
                mask_rgb = image[mask_projh, mask_projw]
                points_rgb[mask_cam] = mask_rgb
        
            # filer points in tracking bbox
            points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

            for tracklet in object_tracklets_vehicle[i]:
                track_id = int(tracklet[0])
                if track_id >= 0:
                    obj_pose_vehicle = np.eye(4)                    
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                    obj_pose_vehicle[:3, 3] = tracklet[1:4]
                    vehicle2local = np.linalg.inv(obj_pose_vehicle)
                    
                    points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                    points_xyz_obj = points_xyz_obj[..., :3]
                    
                    length = object_info[track_id]['length']
                    width = object_info[track_id]['width']
                    height = object_info[track_id]['height']
                    bbox = [[-length/2, -width/2, -height/2], [length/2, width/2, height/2]]
                    obj_corners_3d_local = bbox_to_corner3d(bbox)
                    
                    points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                    points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                    points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                    points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])
        
            points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
            
            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)
            
        initial_num_obj = 20000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    # downsample lidar pointcloud with voxels
                    points_lidar = o3d.geometry.PointCloud()
                    points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                    points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)                                
                elif k.startswith('obj'):  
                    
                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]
                        
                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb
                
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
        
        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        points_xyz_dict['colmap'] = points_colmap_xyz
        points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb
            
        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict

        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue

    return result