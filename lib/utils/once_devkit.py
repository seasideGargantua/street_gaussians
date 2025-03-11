import json
import functools
import os
import os.path as osp
from collections import defaultdict
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import json
from scipy.spatial.transform import Rotation as R

def split_info_loader_helper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        split_file_path = func(*args, **kwargs)
        if not osp.isfile(split_file_path):
            split_list = []
        else:
            split_list = set(map(lambda x: x.strip(), open(split_file_path).readlines()))
        return split_list
    return wrapper


class ONCE(object):
    """
    dataset structure:
    - data_root
        - data
            - seq_id
                - cam01
                - cam03
                - ...
                -
    """
    camera_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
    camera_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']

    def __init__(self, dataset_root, seq_id):
        self.data_root = dataset_root
        self.seq_id = seq_id
        self.load_metadata()

    def load_metadata(self):
        metadata_path = osp.join(self.data_root, self.seq_id, f'{self.seq_id}.json')
        with open(metadata_path, 'r', encoding='utf-8') as file:
            self.metadata = json.load(file)
        self.calib = self.metadata['calib']
        self.meta_info = self.metadata['meta_info']
        frames = self.metadata['frames']
        self.frames = {}
        for frame in frames:
            self.frames[frame['frame_id']]=frame
        
    def get_frame_anno(self):
        split_name = self._find_split_name(self.seq_id)
        frame_info = self.metadata
        if 'annos' in frame_info:
            return frame_info['annos']
        return None

    def load_point_cloud(self, frame_id):
        bin_path = osp.join(self.data_root, self.seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    def load_image(self, frame_id, cam_name):
        cam_path = osp.join(self.data_root, self.seq_id, cam_name, '{}.jpg'.format(frame_id))
        return Image.open(cam_path)

    def load_sky_mask(self, frame_id, cam_name):
        cam_path = osp.join(self.data_root, self.seq_id, 'sky_mask', cam_name, '{}.jpg'.format(frame_id))
        sky_mask = (cv2.imread(cam_path)[..., 0]) > 0.
        return Image.fromarray(sky_mask)

    def load_obj_bound(self, frame_id, cam_name):
        cam_path = osp.join(self.data_root, self.seq_id, 'dynamic_mask', cam_name, '{}.jpg'.format(frame_id))
        obj_bound = (cv2.imread(cam_path)[..., 0]) > 0.
        return Image.fromarray(obj_bound)

    def load_lidar_depth(self, frame_id, cam_name):
        w, h = self.get_WH()
        l2w = self.get_l2w(frame_id)
        w2c = np.linalg.inv(self.get_c2w(frame_id, cam_name))
        ixt = self.get_intr(cam_name)
        points_xyz = self.load_point_cloud(frame_id)[:, :3]
        points_xyz_homo = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
        points_xyz_world = (points_xyz_homo @ l2w.T)[:, :3]
        points_xyz_homo = np.concatenate([points_xyz_world, np.ones_like(points_xyz_world[..., :1])], axis=-1)
        points_xyz_cam = (points_xyz_homo @ w2c.T)[:, :3]
        points_depth = points_xyz_cam[..., 2]
        points_xyz_pixel = points_xyz_cam[..., :3] @ ixt.T
        points_xyz_pixel = points_xyz_pixel / points_xyz_pixel[..., 2:]

        valid_x = np.logical_and(points_xyz_pixel[..., 0] >= 0, points_xyz_pixel[..., 0] < w)
        valid_y = np.logical_and(points_xyz_pixel[..., 1] >= 0, points_xyz_pixel[..., 1] < h)
        valid_z = points_xyz_cam[..., 2] > 0.
        valid_mask = np.logical_and(valid_x, np.logical_and(valid_y, valid_z))
        
        points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
        points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
        points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)

        depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
        u, v = points_coord[:, 0], points_coord[:, 1]
        indices = v * w + u
        np.minimum.at(depth, indices, points_depth[valid_mask])
        depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
        depth = depth.reshape(h, w)
        return depth

    def get_image_path(self, frame_id, cam_name):
        return osp.join(self.data_root, self.seq_id, cam_name, '{}.jpg'.format(frame_id))

    def get_frame_ids(self, cam_name):
        frame_list = os.listdir(osp.join(self.data_root, self.seq_id, cam_name))
        frame_list = [frame_id.strip('.jpg') for frame_id in frame_list]
        return frame_list

    def get_l2w(self, frame_id):
        l2w = np.eye(4)
        pose = self.frames[frame_id]['pose']
        position = np.array(pose[4:])  # First three values: position [x, y, z]
        rotation = np.array(pose[:4])  # Last four values: rotation [qw, qx, qy, qz]
        rotation = R.from_quat(rotation)  # 创建旋转对象
        rotation_matrix = rotation.as_matrix()  # 获取旋转矩阵
        l2w[:3, :3] = rotation_matrix
        l2w[:3, 3] = position
        return l2w

    def get_c2w(self, frame_id, cam_name):
        c2l = np.array(self.calib[cam_name]['cam_to_velo'])
        l2w = np.eye(4)
        pose = self.frames[frame_id]['pose']
        position = np.array(pose[4:])  # First three values: position [x, y, z]
        rotation = np.array(pose[:4])  # Last four values: rotation [qw, qx, qy, qz]
        rotation = R.from_quat(rotation)  # 创建旋转对象
        rotation_matrix = rotation.as_matrix()  # 获取旋转矩阵
        l2w[:3, :3] = rotation_matrix
        l2w[:3, 3] = position
        c2w = np.dot(l2w, c2l)
        return c2w

    def get_c2l(self, cam_name):
        return np.array(self.calib[cam_name]['cam_to_velo'])

    def get_l2c(self, cam_name):
        c2l = np.array(self.calib[cam_name]['cam_to_velo'])
        return np.linalg.inv(c2l)

    def get_intr(self, cam_name):
        return np.array(self.calib[cam_name]['cam_intrinsic'])

    def get_WH(self):
        return self.meta_info['image_size']

    @staticmethod
    def split_point_cloud(points, points_time, rgb, obj_bound, w2c, ixt, w, h):
        obj_bound = np.array(obj_bound)
        # project points to image
        points_xyz = points[:, :3]
        points_xyz_homo = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
        points_xyz_cam = (points_xyz_homo @ w2c.T)[:, :3]
        points_xyz_pixel = points_xyz_cam[..., :3] @ ixt.T
        points_xyz_pixel = points_xyz_pixel / points_xyz_pixel[..., 2:]
        valid_x = np.logical_and(points_xyz_pixel[..., 0] >= 0, points_xyz_pixel[..., 0] < w)
        valid_y = np.logical_and(points_xyz_pixel[..., 1] >= 0, points_xyz_pixel[..., 1] < h)
        valid_z = points_xyz_cam[..., 2] > 0.
        valid_mask = np.logical_and(valid_x, np.logical_and(valid_y, valid_z))
        valid_indices = np.where(valid_mask)[0]
    
        # get valid 2d points
        valid_points_2d = points_xyz_pixel[valid_indices].astype(int)
        
        # get points rgb
        points_rgb = rgb[valid_points_2d[:, 1], valid_points_2d[:, 0]]

        # check if points are in object bound
        mask_values = obj_bound[valid_points_2d[:, 1], valid_points_2d[:, 0]]
        in_mask = mask_values != 0

        dynamic_points = points[valid_indices[in_mask]]
        dynamic_points_time = points_time[valid_indices[in_mask]]
        dynamic_points_rgb = points_rgb[in_mask]

        bkgd_points = points[valid_indices[~in_mask]]
        bkgd_points_time = points_time[valid_indices[~in_mask]]
        bkgd_points_rgb = points_rgb[~in_mask]

        return { 'dynamic_points': dynamic_points,
                 'dynamic_points_time': dynamic_points_time,
                 'dynamic_points_rgb': dynamic_points_rgb,
                 'bkgd_points': bkgd_points,
                 'bkgd_points_time': bkgd_points_time,
                 'bkgd_points_rgb': bkgd_points_rgb, }

    # def undistort_image(self, seq_id, frame_id):
    #     img_list = []
    #     split_name = self._find_split_name(seq_id)
    #     frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
    #     for cam_name in self.__class__.camera_names:
    #         img_buf = self.load_image(seq_id, frame_id, cam_name)
    #         cam_calib = frame_info['calib'][cam_name]
    #         h, w = img_buf.shape[:2]
    #         cv2.getOptimalNewCameraMatrix(cam_calib['cam_intrinsic'],
    #                                       cam_calib['distortion'],
    #                                       (w, h), alpha=0.0, newImgSize=(w, h))
    #         img_list.append(cv2.undistort(img_buf, cam_calib['cam_intrinsic'],
    #                                       cam_calib['distortion'],
    #                                       newCameraMatrix=cam_calib['cam_intrinsic']))
    #     return img_list

    # def undistort_image_v2(self, seq_id, frame_id):
    #     img_list = []
    #     new_cam_intrinsic_dict = dict()
    #     split_name = self._find_split_name(seq_id)
    #     frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
    #     for cam_name in self.__class__.camera_names:
    #         img_buf = self.load_image(seq_id, frame_id, cam_name)
    #         cam_calib = frame_info['calib'][cam_name]
    #         h, w = img_buf.shape[:2]
    #         new_cam_intrinsic, _ = cv2.getOptimalNewCameraMatrix(cam_calib['cam_intrinsic'],
    #                                       cam_calib['distortion'],
    #                                       (w, h), alpha=0.0, newImgSize=(w, h))
    #         img_list.append(cv2.undistort(img_buf, cam_calib['cam_intrinsic'],
    #                                       cam_calib['distortion'],
    #                                       newCameraMatrix=new_cam_intrinsic))
    #         new_cam_intrinsic_dict[cam_name] = new_cam_intrinsic
    #     return img_list, new_cam_intrinsic_dict

    # def project_lidar_to_image(self, seq_id, frame_id):
    #     points = self.load_point_cloud(seq_id, frame_id)

    #     split_name = self._find_split_name(seq_id)
    #     frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
    #     points_img_dict = dict()
    #     img_list, new_cam_intrinsic_dict = self.undistort_image_v2(seq_id, frame_id)
    #     for cam_no, cam_name in enumerate(self.__class__.camera_names):
    #         calib_info = frame_info['calib'][cam_name]
    #         cam_2_velo = calib_info['cam_to_velo']
    #         cam_intri = np.hstack([new_cam_intrinsic_dict[cam_name], np.zeros((3, 1), dtype=np.float32)])
    #         point_xyz = points[:, :3]
    #         points_homo = np.hstack(
    #             [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
    #         points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
    #         mask = points_lidar[:, 2] > 0
    #         points_lidar = points_lidar[mask]
    #         points_img = np.dot(points_lidar, cam_intri.T)
    #         points_img = points_img / points_img[:, [2]]
    #         img_buf = img_list[cam_no]
    #         for point in points_img:
    #             try:
    #                 cv2.circle(img_buf, (int(point[0]), int(point[1])), 2, color=(0, 0, 255), thickness=-1)
    #             except:
    #                 print(int(point[0]), int(point[1]))
    #         points_img_dict[cam_name] = img_buf
    #     return points_img_dict

    @staticmethod
    def rotate_z(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    # def project_boxes_to_image(self, seq_id, frame_id):
    #     split_name = self._find_split_name(seq_id)
    #     if split_name not in ['train', 'val']:
    #         print("seq id {} not in train/val, has no 2d annotations".format(seq_id))
    #         return
    #     frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
    #     img_dict = dict()
    #     img_list, new_cam_intrinsic_dict = self.undistort_image_v2(seq_id, frame_id)
    #     for cam_no, cam_name in enumerate(self.__class__.camera_names):
    #         img_buf = img_list[cam_no]

    #         calib_info = frame_info['calib'][cam_name]
    #         cam_2_velo = calib_info['cam_to_velo']
    #         cam_intri = np.hstack([new_cam_intrinsic_dict[cam_name], np.zeros((3, 1), dtype=np.float32)])

    #         cam_annos_3d = np.array(frame_info['annos']['boxes_3d'])

    #         corners_norm = np.stack(np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(
    #             np.float32)[[0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2], :] - 0.5
    #         corners = np.multiply(cam_annos_3d[:, 3: 6].reshape(-1, 1, 3), corners_norm)
    #         rot_matrix = np.stack(list([np.transpose(self.rotate_z(box[-1])) for box in cam_annos_3d]), axis=0)
    #         corners = np.einsum('nij,njk->nik', corners, rot_matrix) + cam_annos_3d[:, :3].reshape((-1, 1, 3))

    #         for i, corner in enumerate(corners):
    #             points_homo = np.hstack([corner, np.ones(corner.shape[0], dtype=np.float32).reshape((-1, 1))])
    #             points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
    #             mask = points_lidar[:, 2] > 0
    #             points_lidar = points_lidar[mask]
    #             points_img = np.dot(points_lidar, cam_intri.T)
    #             points_img = points_img / points_img[:, [2]]
    #             if points_img.shape[0] != 16:
    #                 continue
    #             for j in range(15):
    #                 cv2.line(img_buf, (int(points_img[j][0]), int(points_img[j][1])), (int(points_img[j+1][0]), int(points_img[j+1][1])), (0, 255, 0), 2, cv2.LINE_AA)

    #         cam_annos_2d = frame_info['annos']['boxes_2d'][cam_name]

    #         for box2d in cam_annos_2d:
    #             box2d = list(map(int, box2d))
    #             if box2d[0] < 0:
    #                 continue
    #             cv2.rectangle(img_buf, tuple(box2d[:2]), tuple(box2d[2:]), (255, 0, 0), 2)

    #         img_dict[cam_name] = img_buf
    #     return img_dict

    # def frame_concat(self, seq_id, frame_id, concat_cnt=0):
    #     """
    #     return new points coordinates according to pose info
    #     :param seq_id:
    #     :param frame_id:
    #     :return:
    #     """
    #     split_name = self._find_split_name(seq_id)

    #     seq_info = getattr(self, '{}_info'.format(split_name))[seq_id]
    #     start_idx = seq_info['frame_list'].index(frame_id)
    #     points_list = []
    #     translation_r = None
    #     try:
    #         for i in range(start_idx, start_idx + concat_cnt + 1):
    #             current_frame_id = seq_info['frame_list'][i]
    #             frame_info = seq_info[current_frame_id]
    #             transform_data = frame_info['pose']
    
    #             points = self.load_point_cloud(seq_id, current_frame_id)
    #             points_xyz = points[:, :3]
    
    #             rotation = Rotation.from_quat(transform_data[:4]).as_matrix()
    #             translation = np.array(transform_data[4:]).transpose()
    #             points_xyz = np.dot(points_xyz, rotation.T)
    #             points_xyz = points_xyz + translation
    #             if i == start_idx:
    #                 translation_r = translation
    #             points_xyz = points_xyz - translation_r
    #             points_list.append(np.hstack([points_xyz, points[:, 3:]]))
    #     except ValueError:
    #         print('warning: part of the frames have no available pose information, return first frame point instead')
    #         points = self.load_point_cloud(seq_id, seq_info['frame_list'][start_idx])
    #         points_list.append(points)
    #         return points_list
    #     return points_list


if __name__ == '__main__':
    dataset = ONCE('/root')
    for seq_id, frame_id in [('000092', '1616442892300')]:
        img_buf_dict = dataset.project_boxes_to_image(seq_id, frame_id)
        for cam_name, img_buf in img_buf_dict.items():
            cv2.imwrite('images/box_project_{}_{}.jpg'.format(cam_name, frame_id), cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
        img_buf_dict = dataset.project_lidar_to_image(seq_id, frame_id)
        for cam_name, img_buf in img_buf_dict.items():
            cv2.imwrite('images/lidar_project_{}_{}.jpg'.format(cam_name, frame_id), cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))