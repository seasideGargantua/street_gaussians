import pandas
import json
from scipy.spatial.transform import Rotation
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
from lib.utils.general_utils import parse_one_obj_info_numpy
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points
from lib.utils.data_utils import get_val_frames, get_rays, sphere_intersection
from lib.utils.colmap_utils import read_points3D_binary

# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir, start_frame, end_frame):
    # import ipdb; ipdb.set_trace()
    infos_file_name = datadir.split("/")[-1] + '.json'
    infos_dir = os.path.join(datadir.replace("3D_images", "3D_infos"), infos_file_name)
    import json 
    
    with open(infos_dir, 'r') as file:
        anno_data = json.load(file)
    
    #TODO: 获取 cam3的内外参
    intrinsics = []
    extrinsics = []    
    lidar2world = []
    
    # Get the intrinsic parameters for cam03
    cam03_intrinsic = anno_data["calib"]["cam03"]["cam_intrinsic"]
    cam03_extrinsic = anno_data["calib"]["cam03"]["cam_to_velo"]  # Camera-to-Velo matrix

    # Extracting the intrinsics (3x3 matrix)
    # intrinsics = np.array(cam03_intrinsic)  # Camera intrinsics matrix for cam03

    # Convert the extrinsic parameters (cam_to_velo) into a numpy matrix (4x4 matrix)
    cam_to_velo = np.array(cam03_extrinsic)  # Camera extrinsic matrix for cam03 (transform to velo frame)
    
    # Iterate over start frame and end frame
    cam03_c2w_matrices = []

    for frame_idx in range(start_frame, end_frame+1):  # Use provided frame range
        frame_pose = anno_data["frames"][frame_idx]["pose"] # Frame pose for each frame
        
        frame_position = np.array(frame_pose[4:])  # First three values: position [x, y, z]
        frame_rotation = np.array(frame_pose[:4])  # Last four values: rotation [qw, qx, qy, qz] 
        from scipy.spatial.transform import Rotation as R
        # 使用 scipy 的 Rotation 类将四元组转换为旋转矩阵
        rotation = R.from_quat(frame_rotation)  # 创建旋转对象
        rotation_matrix = rotation.as_matrix()  # 获取旋转矩阵

        l2w_matrix = np.eye(4)
        l2w_matrix[:3, :3] = rotation_matrix
        l2w_matrix[:3, 3] = frame_position

        T_velo_to_cam03 = np.linalg.inv(cam_to_velo)
        # Now, we combine the world pose (C2W) with cam_to_velo
        # The world-to-camera (C2W) matrix combined with cam_to_velo will give us the Camera-to-World (C2W) for cam03

        cam03_c2w_frame = np.dot(l2w_matrix, cam_to_velo)
        # cam03_c2w_frame = c2w_matrix
        # Store the C2W matrix for cam03 for this frame
        cam03_c2w_matrices.append(cam03_c2w_frame)
        # cam03_c2w_frame[:3, 1:3] *= -1
        # c2w_matrix[:3, 1:3] *= -1
        extrinsics.append(cam03_c2w_frame)
        intrinsics.append(np.array(cam03_intrinsic))
        lidar2world.append(l2w_matrix)
    
    return intrinsics, extrinsics, lidar2world


def generate_dataparser_outputs(
        datadir,
        obj_maskdir,
        selected_frames=None, 
        use_tracker=False,
        build_pointcloud=False, 
        build_moving_vehicle_bound=False,        
        building_depth=True,
    ):

    # import ipdb; ipdb.set_trace()
    
    image_dir = os.path.join(datadir, 'cam03')
    
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.jpg')))
    
    num_frames_all = len(image_filenames_all)
    # num_cameras = len(cameras)
    
    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    image_filenames = image_filenames_all[start_frame: end_frame + 1]
    frames = list(range(start_frame, end_frame + 1))
    frames_idx = list(range(len(frames))) 
    cams = [3] * len(frames)
    # load calibration and ego pose
    ixts, exts, lidar2worlds = load_camera_info(datadir, start_frame, end_frame)    
    
    result = dict()
    result['num_frames'] = len(frames)
    result['ixts'] = ixts
    result['c2ws'] = exts
    result['poses'] = lidar2worlds
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['obj_tracklets'] = tracklet
    result['obj_info'] = obj_meta
    result['scene_classes'] = scene_classes
    result['scene_objects'] = scene_objects
    result['cams_timestamps'] = np.array(frames)
    result['tracklet_timestamps'] = np.array(frames)
    
    obj_bounds = []
    if os.path.exists(obj_maskdir):
        obj_mask_filenames = sorted(glob(os.path.join(obj_maskdir, 'cam03/*.jpg')))
        obj_mask_filenames = obj_mask_filenames[start_frame: end_frame + 1]
        for obj_mask_filename in obj_mask_filenames:
            frame_idx = int(os.path.basename(obj_mask_filename).split('.')[0]) - start_frame
            obj_mask = cv2.imread(obj_mask_filename, cv2.IMREAD_GRAYSCALE) / 255.
            obj_mask = obj_mask.astype(np.bool)
            obj_bounds.append(obj_mask)
        result['obj_bounds'] = obj_bounds

    building_depth = not os.path.exists(os.path.join(cfg.model_path, 'lidar_depth'))
    if building_depth:
        from lib.utils.img_utils import visualize_depth_numpy

        print('building depth')
        depth_dir = os.path.join(cfg.model_path, 'lidar_depth')
        os.makedirs(depth_dir, exist_ok=True)

        print('initialize from lidar pointcloud')

        points_dict = {}
        frame_ids = []
        for image_path in image_filenames:
            # 通过修改 image_path 构造 .bin 文件路径
            bin_path = image_path.replace("3D_images", "3D_lidars").replace("cam03", "lidar_roof").replace(".jpg", ".bin")

            if not os.path.exists(bin_path):
                print(f"Warning: {bin_path} 不存在，跳过该帧！")
                continue
            # 读取 .bin 文件，转换为 NumPy 数组
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (N, 4) -> [x, y, z, intensity]
            # 生成帧 ID（你可以根据数据集的文件命名规则进行调整）
            frame_id = os.path.basename(bin_path).replace(".bin", "")
            frame_ids.append(frame_id)
            # 存入字典
            points_dict[frame_id] = points
            

        i = 0
        for image_path in image_filenames:
            frame_id = os.path.basename(image_path).replace(".jpg", "")
            image = cv2.imread(image_path)[..., [2, 1, 0]] / 255.
            h, w = image.shape[:2]
            points_xyz = points_dict[frame_id][..., :3]

            points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
            l2w_matrix = lidar2worlds[i]
            world_points_homo = np.dot(l2w_matrix, points_xyz.T).T
            points_xyz = world_points_homo[:, :3]
            
            points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
            
            # filter points with camera frustum
            c2w = exts[i]
            ixt = ixts[i]            
            w2c = np.linalg.inv(c2w)
                        
            points_xyz_cam = points_xyz @ w2c.T
            points_depth = points_xyz_cam[..., 2]
            points_xyz_pixel = points_xyz_cam[..., :3] @ ixt.T
            points_xyz_pixel = points_xyz_pixel / points_xyz_pixel[..., 2:]
            
            valid_x = np.logical_and(points_xyz_pixel[..., 0] >= 0, points_xyz_pixel[..., 0] < w)
            valid_y = np.logical_and(points_xyz_pixel[..., 1] >= 0, points_xyz_pixel[..., 1] < h)
            valid_z = points_xyz_cam[..., 2] > 0.
            valid_mask = np.logical_and(valid_x, np.logical_and(valid_y, valid_z))
            
            points_xyz = points_xyz[valid_mask]
            points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
            points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
            points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)
            
            depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
            u, v = points_coord[:, 0], points_coord[:, 1]
            indices = v * w + u
            np.minimum.at(depth, indices, points_depth[valid_mask])
            depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
            depth = depth.reshape(h, w)
            # import ipdb; ipdb.set_trace()
            depth_vis, _ = visualize_depth_numpy(depth)
            depth_path = os.path.join(depth_dir, f'{os.path.basename(image_filenames[i]).split(".")[0]}.npy')
            np.save(depth_path, depth)
            depth_vis_path = os.path.join(depth_dir, f'{os.path.basename(image_filenames[i]).split(".")[0]}.png')
            cv2.imwrite(depth_vis_path, depth_vis)
            i += 1
    
    # import ipdb; ipdb.set_trace()
    build_pointcloud = not os.path.exists(os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply'))

    infos_file_name = datadir.split("/")[-1] + '.json'
    infos_dir = os.path.join(datadir.replace("3D_images", "3D_infos"), infos_file_name)
    import json 
    
    with open(infos_dir, 'r') as file:
        anno_data = json.load(file)

    
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)

        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        
        split_test = cfg.data.get('split_test', -1)
        split_train = cfg.data.get('split_train', -1)
        num_frames = len(image_filenames)
        train_frames, test_frames = get_val_frames(
            num_frames, 
            test_every=split_test if split_test > 0 else None,
            train_every=split_train if split_train > 0 else None,
        )
        
        print('initialize from sfm pointcloud')
        pointcloud_colmap = os.path.join(f'{cfg.model_path}/colmap/triangulated/sparse/model/points3D.bin')
        if not os.path.exists(pointcloud_colmap):
            from script.Dream.colmap_Dream import run_colmap_Dream
            run_colmap_Dream(result)
        assert os.path.exists(pointcloud_colmap)
        points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(pointcloud_colmap)
        points_colmap_rgb = points_colmap_rgb / 255
        
        points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
        for i, frame in enumerate(range(start_frame, end_frame+1)):
            if frame not in train_frames:
                continue
            camera_posistion = exts[i][:3, 3]
            plane_dist = np.linalg.norm(points_colmap_xyz[:, :2] - camera_posistion[:2], axis=-1)
            mask1 = plane_dist < 20
            mask2 = points_colmap_xyz[:, 2] < camera_posistion[2] + 2
            mask = np.logical_and(mask1, mask2)
            points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)
            
        points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
        points_colmap_rgb = points_colmap_rgb[points_colmap_mask]
        print('colmap pointcloud size: ', points_colmap_xyz.shape[0]) 
        print('colmap pointcloud size: ', points_colmap_xyz.shape[0])
        
        # import ipdb; ipdb.set_trace()
        
        print('initialize from lidar pointcloud')

        points_dict = {}
        frame_ids = []
        i = 0
        for image_path in image_filenames:
            # 通过修改 image_path 构造 .bin 文件路径
            bin_path = image_path.replace("3D_images", "3D_lidars").replace("cam03", "lidar_roof").replace(".jpg", ".bin")

            if not os.path.exists(bin_path):
                print(f"Warning: {bin_path} 不存在，跳过该帧！")
                continue
            # 读取 .bin 文件，转换为 NumPy 数组
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (N, 4) -> [x, y, z, intensity]
            # 生成帧 ID（你可以根据数据集的文件命名规则进行调整）
            frame_id = os.path.basename(bin_path).replace(".bin", "")
            frame_ids.append(frame_id)
            # 存入字典
            # points_dict[frame_id] = points
            points = points[..., :3]
            points_xyz = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
            l2w_matrix = lidar2worlds[i]
            i += 1
            world_points_homo = np.dot(l2w_matrix, points_xyz.T).T
            points_xyz = world_points_homo[:, :3]
            points_dict[frame_id] = points_xyz
        

        # pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        # points_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        if obj_meta is not None:
            for obj_meta_ in obj_meta.keys():
                track_id = int(obj_meta_)
                if track_id >= 0:
                    points_xyz_dict[f'obj_{track_id}'] = []
                    points_rgb_dict[f'obj_{track_id}'] = []
        
        for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
            if frames_idx[i] not in train_frames:
                continue
            image_path = image_filenames[i]
            image = cv2.imread(image_path)[..., [2, 1, 0]] / 255.
            h, w = image.shape[:2]
            frame_id = frame_ids[i]
            points_xyz = points_dict[frame_id][..., :3]
            points_xyz = points_xyz
            points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
            
            # filter points with camera frustum
            c2w = exts[i]
            ixt = ixts[i]            
            w2c = np.linalg.inv(c2w)
                        
            points_xyz_cam = points_xyz @ w2c.T
            points_xyz_pixel = points_xyz_cam[..., :3] @ ixt.T
            points_xyz_pixel = points_xyz_pixel / points_xyz_pixel[..., 2:]
            
            valid_x = np.logical_and(points_xyz_pixel[..., 0] >= 0, points_xyz_pixel[..., 0] < w)
            valid_y = np.logical_and(points_xyz_pixel[..., 1] >= 0, points_xyz_pixel[..., 1] < h)
            valid_z = points_xyz_cam[..., 2] > 0.
            valid_mask = np.logical_and(valid_x, np.logical_and(valid_y, valid_z))
            
            points_xyz = points_xyz[valid_mask]
            points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
            points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
            points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)
                                                
            points_rgb = image[points_coord[..., 1], points_coord[..., 0]]

            # filer points in tracking bbox
            points_xyz_obj_mask = np.zeros(points_xyz.shape[0], dtype=np.bool_)
            if obj_data is not None:
                for obj_data_ in obj_data[i]:
                    # import ipdb; ipdb.set_trace()
                    track_id = int(obj_data_[7])
                    if track_id >= 0:
                        obj_pose, obj_corners3d = parse_one_obj_info_numpy(obj_data_)
                        world2obj = np.linalg.inv(obj_pose)
                        points_xyz_obj = points_xyz @ world2obj.T
                        points_xyz_obj = points_xyz_obj[..., :3]
                        obj_corners3d = obj_corners3d @ world2obj.T[:3, :3] + world2obj[:3, 3]
                        points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners3d)
                        points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                        
                        points_xyz_dict[f'obj_{track_id}'].append(points_xyz_obj[points_xyz_inbbox])
                        points_rgb_dict[f'obj_{track_id}'].append(points_rgb[points_xyz_inbbox])
            
            points_lidar_xyz = points_xyz[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
            
            points_xyz_dict['bkgd'].append(points_xyz[..., :3])
            points_rgb_dict['bkgd'].append(points_rgb)
            
        initial_num_bkgd = cfg.data.get('initial_num_bkgd', 100000)
        initial_num_obj = cfg.data.get('initial_num_obj', 20000)
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
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.25)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    # downsample_points_lidar = points_lidar
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
                
        points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0) 
        points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
          
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