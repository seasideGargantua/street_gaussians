import pandas
import json
from scipy.spatial.transform import Rotation #把四元组转换为旋转矩阵
import os
import numpy as np
import cv2
import open3d as o3d
import torch
import math
from glob import glob
from tqdm import tqdm
from lib.config import cfg
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import storePly, get_Sphere_Norm
from lib.utils.general_utils import parse_one_obj_info_numpy
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points
from lib.utils.data_utils import get_val_frames, get_rays, sphere_intersection
from lib.utils.colmap_utils import read_points3D_binary


def build_frame_name_to_idx(datadir, selected_frames):
    start_frame, end_frame = selected_frames[0], selected_frames[1]
    
    # 获取 image_dir 路径
    image_dir = os.path.join(datadir, 'cam03')
    
    # 获取所有图片文件名
    image_names = sorted(os.listdir(image_dir))  # 按照文件名排序，假设文件名为时间戳格式
    
    # 构建 frame_name2frame_idx 字典
    frame_name2frame_idx = {}
    
    # 只考虑 selected_frames 范围内的文件名
    for idx, image_name in enumerate(image_names[start_frame:end_frame + 1]):
        # 去掉文件扩展名
        frame_name = image_name.split('.')[0]
        # 将文件名映射到对应的索引
        frame_name2frame_idx[int(frame_name)] = idx
    
    return frame_name2frame_idx


def quaternion_from_yaw(yaw_angle):
    """
    Create a quaternion for a yaw rotation around the z-axis.

    Parameters:
    - yaw_angle (float): Yaw angle in radians.

    Returns:
    - np.array: Quaternion [a, b, c, d].
    """
    # Calculate half angle
    half_angle = 0.5 * yaw_angle

    # Quaternion components
    a = np.cos(half_angle)
    b = 0.0
    c = 0.0
    d = np.sin(half_angle)

    return np.array([a, b, c, d]).astype(np.float64)

_track2label = {"car": 1, "cyclist": 2, "pedestrian": 3, "truck": 4,} #: update to semantic label
def get_obj_pose_tracking(datadir, selected_frames):
    
    # import ipdb; ipdb.set_trace()
    objects_meta_Once = dict()

    objects_meta = {}
    tracklets_ls = []
    
    tracklet_path = glob(os.path.join(datadir.replace("3D_images", "3D_infos"), '*.txt'))[0]
    json_path = glob(os.path.join(datadir.replace("3D_images", "3D_infos"), '*.json'))[0]
    import json 
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    f = open(tracklet_path, 'r')
    tracklets_str = f.read().splitlines()
    # import ipdb; ipdb.set_trace()
    tracklets_str = tracklets_str[1:] #tracklets_str[0]是标签，tracklets_str[1:]是数据。    
    start_frame, end_frame = selected_frames[0], selected_frames[1]
    
    image_dir = os.path.join(datadir, 'cam03')
    
    frame_name2frame_idx = build_frame_name_to_idx(datadir, selected_frames)
    
    n_obj_in_frame = np.zeros(end_frame+1) 

    # Initialize lists and sets
    obj_data = []
    obj_meta = []
    scene_objects = set()
    scene_classes = set()
       
    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        
        frame_id = int(tracklet[0])
        # 遍历 json_data["frames"] 查找匹配的 frame_id
        for frame_data in json_data["frames"]:
            if frame_data["frame_id"] == str(frame_id):  # 注意 json 中的 frame_id 是字符串
                frame_pose = frame_data["pose"]
                print("Pose for frame_id", frame_id, ":", frame_pose)
                break
        
        frame_position = np.array(frame_pose[4:])  # First three values: position [x, y, z]
        frame_rotation = np.array(frame_pose[:4])  # Last four values: rotation [qw, qx, qy, qz] 
        from scipy.spatial.transform import Rotation as R
        # 使用 scipy 的 Rotation 类将四元组转换为旋转矩阵
        rotation = R.from_quat(frame_rotation)  # 创建旋转对象
        rotation_matrix = rotation.as_matrix()  # 获取旋转矩阵

        l2w_matrix = np.eye(4)
        l2w_matrix[:3, :3] = rotation_matrix
        l2w_matrix[:3, 3] = frame_position
        
        track_id = int(tracklet[1])
        object_class = tracklet[2]
        center_x, center_y, center_z = float(tracklet[3]), float(tracklet[4]), float(tracklet[5]) 
        lidar_xyz = np.array([center_x, center_y, center_z])
        lidar_xyz = np.concatenate([lidar_xyz, np.ones_like(lidar_xyz[..., :1])], axis=-1)
        world_xyz = np.dot(l2w_matrix, lidar_xyz.T).T
        world_xyz = world_xyz[:3]        
        center_x, center_y, center_z = world_xyz
        
        #TODO xyz需要从lidar坐标系转换到世界坐标系
        length, width, height = float(tracklet[6]), float(tracklet[7]), float(tracklet[8]) 
        quaternion = quaternion_from_yaw(float(tracklet[9]))
        
        center_x = np.array([center_x])
        center_y = np.array([center_y])
        center_z = np.array([center_z])
        pose3d = np.concatenate([center_x, center_y, center_z,quaternion])
        
        obj_type = np.array([_track2label[object_class]])
        obj = np.concatenate([np.array([frame_id, track_id]), obj_type, np.array([length, width, height]), pose3d])
     
        if track_id not in objects_meta_Once:
            objects_meta_Once[track_id] = np.array([track_id, obj_type, length, width, height])
        
        rw, rx, ry, rz = quaternion
        tr_array = np.array([frame_id, track_id, obj_type.item(), height, width, length, center_x.item(), center_y.item(), center_z.item(), rw, rx, ry, rz, 0])      
        tracklets_ls.append(tr_array)
        
        frame_idx_ = frame_name2frame_idx[frame_id]
        
        n_obj_in_frame[frame_idx_] += 1

    tracklets_array = np.array(tracklets_ls)

    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    
    num_frames = end_frame - start_frame + 1
    visible_objects = np.ones([num_frames, max_obj_per_frame, 13]) * -1.0


    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        
        obj_type = np.array(objects_meta_Once[track_id][1])
        dim = objects_meta_Once[track_id][-3:].astype(np.float32)

        if track_id not in objects_meta:
            
            objects_meta[track_id] = np.concatenate(
                [
                    np.array([track_id]).astype(np.float32),
                    objects_meta_Once[track_id][2:].astype(np.float64),
                    np.array(objects_meta_Once[track_id][1]).astype(np.float64),
                ]
            )
                    
        pose3d = tracklet[6:13]
        
        
        obj = np.concatenate([np.array([frame_id, track_id]), obj_type, dim, pose3d])
        frame_idx_ = frame_name2frame_idx[frame_id]

        obj_column = np.argwhere(visible_objects[frame_idx_, :, 0] < 0).min()

        visible_objects[frame_idx_, obj_column] = obj    
    

    obj_state = visible_objects[:, :, [1, 2]] # [track_id, class_id]
    obj_pose = visible_objects[:, :, 6:] #center_x, center_y, center_z, rw, rx, ry, rz
    obj_track_id = obj_state[..., 0][..., None]
    obj_meta_ls = list(objects_meta.values()) # object_id, length, width, height, class_id
    obj_meta_ls.insert(0, np.zeros_like(obj_meta_ls[0]))
    obj_meta_ls[0][0] = -1


    row_to_track_id = np.concatenate(
        [
            np.linspace(0, len(objects_meta.values()), len(objects_meta.values()) + 1)[:, None],
            np.array(obj_meta_ls)[:, 0][:, None],
        ],
        axis=1,
    ).astype(np.int32)
    # [n_frames, n_max_obj]
    track_row = np.zeros_like(obj_track_id)
    
    scene_objects = []
    scene_classes = list(np.unique(np.array(obj_meta_ls)[..., 4]))
    for i, frame_objects in enumerate(obj_track_id):
        for j, camera_objects in enumerate(frame_objects):
            track_row[i, j] = np.argwhere(row_to_track_id[:, 1] == camera_objects)
            if camera_objects >= 0 and not camera_objects in scene_objects:
                # print(camera_objects, "in this scene")
                scene_objects.append(camera_objects)


    box_scale = 2
    print('box scale: ', box_scale)
    obj_meta_ls = [
        (obj * np.array([1.0, box_scale, box_scale, box_scale, 1.0])).astype(np.float32)
        for obj in obj_meta_ls
    ]  # [n_obj, [track_id, length, width, height, class_id]]
    obj_meta = np.array(obj_meta_ls).astype(np.float32)
     
    # [n_frames, n_obj, [track_id, length, width, height, class_id]]
    obj_meta_idx = track_row.squeeze(-1).astype(np.int32)
    
    frames = list(range(start_frame, end_frame + 1))
    obj_frame_range = np.zeros([len(obj_meta), 2]).astype(np.int32)
    for i in range(len(obj_meta)):
        if i == 0:
            continue
        
        obj_frame_idx = np.argwhere(obj_meta_idx == i)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        frames_numpy = np.array(frames).astype(np.int32)
        obj_frames = frames_numpy[obj_frame_idx]

        obj_frame_range[i, 0] = np.min(obj_frames)
        obj_frame_range[i, 1] = np.max(obj_frames)
    
    batch_obj_meta = obj_meta[obj_meta_idx] 
    
    #obj_pose: center_x, center_y, center_z, rw, rx, ry, rz
    #batch_obj_meta: track_id, length, width, height, type
    obj_data = np.concatenate([obj_pose, batch_obj_meta], axis=-1)
    data_len = obj_data.shape[0]
    meta_data = np.array([[num_frames, max_obj_per_frame]] * data_len)[:,None,:].repeat(max_obj_per_frame, axis=1)
    batch_trackid = batch_obj_meta[... ,0]
    # __import__('ipdb').set_trace()
    meta_data = np.concatenate([meta_data, batch_trackid[..., None]], axis=-1)
    tracklet = np.concatenate([meta_data, obj_pose], axis=-1)

    # track_id + length + width + height + class_id + start frame + end frame
    obj_meta = np.concatenate([obj_meta, obj_frame_range], axis=-1)
    # import ipdb; ipdb.set_trace()
    obj_infos = {}
    for i in range(obj_meta.shape[0]):
        track_id = obj_meta[i][0]
        length = obj_meta[i][1]
        width = obj_meta[i][2]
        height = obj_meta[i][3]
        class_id = obj_meta[i][4]
        start_frame = obj_meta[i][5]
        end_frame = obj_meta[i][6]

        obj_infos[track_id] = {
            "track_id": track_id,
            "length": length,
            "width": width,
            "height": height,
            "class": class_id,
            "class_label": 'car',
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    return tracklet, obj_infos, scene_classes, scene_objects, obj_data
    
    
    # max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())

    # num_frames = end_frame - start_frame + 1
    # visible_objects = np.ones([num_frames, max_obj_per_frame, 13]) * -1.0 #初始化的时候都是-1


    
    
    # obj_data = np.empty((len(exts), 0, 12))
    # obj_meta = np.array([[-1, 0, 0, 0, 0, 0, 0]])
    # scene_classes = [0.0]
    # scene_objects = []
    
    # return obj_data, obj_meta, scene_classes, scene_objects


def get_obj_pose_tracking_empty(datadir, exts):
    obj_data = np.empty((len(exts), 0, 12))
    obj_meta = np.array([[-1, 0, 0, 0, 0, 0, 0]])
    scene_classes = [0.0]
    scene_objects = []
    
    return obj_data, obj_meta, scene_classes, scene_objects



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

    if cfg.data.get('skip_obj', False):
        tracklet, obj_data, obj_meta, scene_classes, scene_objects = None, None, None, None, None
    else:
        tracklet, obj_meta, scene_classes, scene_objects, obj_data = get_obj_pose_tracking(datadir, selected_frames)
    
    #obj_data  (61, 7, 12)
    #obj_meta  (13, 7) [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #     0.00000000e+00],
    #scene_classes [0.0, 1.0, 3.0, 4.0, 5.0, 7.0]
    #scene_objects [array([2.]), array([9.]), array([14.]), array([53.]), array([15.]), array([55.]), array([234.]), array([932.]), array([1408.]), array([3025.]), array([239.]), array([2125.])]
    
    
    
    # import ipdb; ipdb.set_trace()
    
    
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
                                        
                    # if len(points_lidar_xyz) > initial_num_bkgd:
                    #     random_indices = np.random.choice(len(points_lidar_xyz), initial_num_bkgd, replace=False)
                    #     points_lidar_xyz = points_lidar_xyz[random_indices]
                    #     points_lidar_rgb = points_lidar_rgb[random_indices]
                                
                elif k.startswith('obj'):
                    # points_obj = o3d.geometry.PointCloud()
                    # points_obj.points = o3d.utility.Vector3dVector(points_xyz)
                    # points_obj.colors = o3d.utility.Vector3dVector(points_rgb)
                    # downsample_points_lidar = points_obj.voxel_down_sample(voxel_size=0.05)
                    # points_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    # points_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)  
                    
                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]
                        
                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb
                
                else:
                    raise NotImplementedError()
                
        # Get sphere center and radius
        # lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        # sphere_center = lidar_sphere_normalization['center']
        # sphere_radius = lidar_sphere_normalization['radius']

        # combine SfM pointcloud with LiDAR pointcloud
        # points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
        # mask = points_colmap_dist < 2 * sphere_radius
        # points_colmap_xyz = points_colmap_xyz[mask]
        # points_colmap_rgb = points_colmap_rgb[mask]
        
        # assert cfg.data.use_colmap or cfg.data.use_lidar
        # if cfg.data.use_colmap and not cfg.data.use_lidar:
        #     points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(pointcloud_colmap)
        #     points_colmap_rgb = points_colmap_rgb / 255
            
        #     if len(points_colmap_xyz) < 1000:
        #         print(f'Sfm fails for background with {len(points_colmap_xyz)} points, still using LiDAR point cloud')
        #         points_bkgd_xyz = points_lidar_xyz
        #         points_bkgd_rgb = points_lidar_rgb
        #     else:
        #         points_bkgd_xyz = points_colmap_xyz
        #         points_bkgd_rgb = points_colmap_rgb
            
        #     print('No pointcloud for object, will perform random initialization for object pointcloud')            
        #     # no pointcloud for object
        #     points_xyz_dict.clear()            
        # elif cfg.data.use_lidar and not cfg.data.use_colmap:
        #     points_bkgd_xyz = points_lidar_xyz
        #     points_bkgd_rgb = points_lidar_rgb
        # else:
        #     points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0) 
        #     points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
        
        points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0) 
        points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
          
        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        points_xyz_dict['colmap'] = points_colmap_xyz
        points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb
        
        # Sample sky point cloud        
        # background_sphere_points = 50000
        # background_sphere_distance = 2.5    
        
        # if cfg.model.nsg.get('include_sky', False):
        #     sky_mask_dir = os.path.join(datadir, 'sky_mask')  
        #     assert os.path.exists(sky_mask_dir)
        #     points_xyz_sky_mask = []
        #     points_rgb_sky_mask = []
        #     num_samples = background_sphere_points // len(train_frames)
        #     print('sample points from sky mask for background sphere')
        #     for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
        #         if frames_idx[i] not in train_frames:
        #             continue
                
        #         image_path = image_filenames[i]
        #         image = cv2.imread(image_path)[..., [2, 1, 0]] / 255.
        #         H, W, _ = image.shape
                
        #         sky_mask_path = os.path.join(sky_mask_dir,  os.path.basename(image_path))
        #         sky_mask = (cv2.imread(sky_mask_path)[..., 0] > 0).reshape(-1)
        #         sky_indices = np.argwhere(sky_mask == True)[..., 0]
                
        #         if len(sky_indices) == 0:
        #             continue
        #         elif len(sky_indices) > num_samples:
        #             random_indices = np.random.choice(len(sky_indices), num_samples, replace=False)
        #             sky_indices = sky_indices[random_indices]
                
        #         K = ixts[i]
        #         w2c = np.linalg.inv(exts[i])
        #         R, T = w2c[:3, :3], w2c[:3, 3]
        #         rays_o, rays_d = get_rays(H, W, K, R, T)
        #         rays_o = rays_o.reshape(-1, 3)[sky_indices]
        #         rays_d = rays_d.reshape(-1, 3)[sky_indices]

        #         p_sphere = sphere_intersection(rays_o, rays_d, sphere_center, sphere_radius * background_sphere_distance)
        #         points_xyz_sky_mask.append(p_sphere)
                
        #         pixel_value = image.reshape(-1, 3)[sky_indices]
        #         points_rgb_sky_mask.append(pixel_value)
                
        #     points_xyz_sky_mask = np.concatenate(points_xyz_sky_mask, axis=0)
        #     points_rgb_sky_mask = np.concatenate(points_rgb_sky_mask, axis=0)    
        #     points_xyz_dict['sky'] = points_xyz_sky_mask
        #     points_rgb_dict['sky'] = points_rgb_sky_mask

        # elif cfg.data.get('add_background_sphere', False):            
        #     print('sample random points for background sphere')

        #     # Random sample points on the sphere
        #     samples = np.arange(background_sphere_points)
        #     y = 1.0 - samples / float(background_sphere_points) * 2 # y in [-1, 1]
        #     radius = np.sqrt(1 - y * y) # radius at y
        #     phi = math.pi * (math.sqrt(5.) - 1.) # golden angle in radians
        #     theta = phi * samples # golden angle increment
        #     x = np.cos(theta) * radius
        #     z = np.sin(theta) * radius
        #     unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
            
        #     points_xyz_sky_random = (unit_sphere_points * sphere_center * background_sphere_distance) + sphere_radius
        #     points_rgb_sky_random = np.asarray(np.random.random(points_xyz_sky_random.shape) * 255, dtype=np.uint8)
            
        #     points_xyz_dict['sky'] = points_xyz_sky_random
        #     points_rgb_dict['sky'] = points_rgb_sky_random
        # else:
        #     pass
                
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