import os 
import numpy as np
from scipy import interpolate
import cv2
import point_cloud_utils as pcu
from pyvr import volume_render
import gputools
import argparse 
from tqdm import tqdm


class Dataset(object):

    def __init__(self, us_dir, seg_dir, probe_pose_path, img_to_probe_path):
        self.us_dir = us_dir
        self.seg_dir = seg_dir
        img_names = sorted(os.listdir(us_dir))

        data = np.load(open(probe_pose_path, 'rb'))
        t_pose, probe_pose = data['t'], data['pose']
        data = np.load(open(img_to_probe_path, 'rb'))
        self.R_img_to_probe = data['R_img_to_probe']
        self.t_img_to_probe = data['t_img_to_probe']
        self.scale_xy = data['s']

        t_us = np.array([float(img_name[:-4]) / 1e+9 for img_name in img_names])
        t_start = max(t_pose.min(), t_us.min())
        t_end = min(t_pose.max(), t_us.max())

        keep = np.logical_and(t_us >= t_start, t_us <= t_end)
        self.img_names = np.array(img_names)[keep]
        self.time_us = t_us[keep]

        keep = np.logical_and(t_pose >= t_start, t_pose <= t_end)
        t_pose, probe_pose = t_pose[keep], probe_pose[keep]

        f = interpolate.interp1d(t_pose, probe_pose, axis=0, fill_value='extrapolate')
        probe_pose = f(self.time_us)
        probe_pose[:, 3, 3] = 1
        u, s, vh = np.linalg.svd(probe_pose[:, :3, :3]) 
        probe_pose[:, :3, :3] = np.matmul(u, vh)
        probe_pose = np.matmul(np.linalg.inv(probe_pose[0]), probe_pose)
        self.R_probe, self.t_probe = probe_pose[:, :3, :3], probe_pose[:, :3, 3]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        us_img = cv2.imread(os.path.join(self.us_dir, self.img_names[idx]))[..., 0] / 255.0
        seg_name = self.img_names[idx].split('.')[0] + '.npy'
        seg = np.load(open(os.path.join(self.seg_dir, seg_name), 'rb'))
        R_us_to_cam = self.R_probe[idx].dot(self.R_img_to_probe)
        t_us_to_cam = self.R_probe[idx].dot(self.t_img_to_probe) + self.t_probe[idx]
        return us_img, seg, R_us_to_cam, t_us_to_cam, self.scale_xy


def img_to_cam_pc(img, R_us_to_cam, t_us_to_cam, scale_xy):
    assert len(img.shape) == 2
    img_h, img_w = img.shape
    xs, ys = np.meshgrid(np.arange(img_w), np.arange(img_h))
    vs = img[ys, xs]
    img_pc = np.concatenate([
        xs.reshape(-1, 1) * scale_xy[0], ys.reshape(-1, 1) * scale_xy[1], np.zeros((img_h*img_w, 1))], axis=1)
    cam_pc = img_pc.dot(R_us_to_cam.T) + t_us_to_cam
    cam_pc = np.concatenate([cam_pc, vs.reshape(-1, 1)], axis=1)
    return cam_pc


def center_and_reduce(pc, center, max_bound, min_brightness):
    pc = pc[pc[:, 3] > min_brightness]
    pc[:, :3] = pc[:, :3] - center
    x_in_bound = np.logical_and(pc[:, 0] > -max_bound / 2, pc[:, 0] < max_bound / 2)
    y_in_bound = np.logical_and(pc[:, 1] > -max_bound / 2, pc[:, 1] < max_bound / 2)
    z_in_bound = np.logical_and(pc[:, 2] > -max_bound / 2, pc[:, 2] < max_bound / 2)
    xyz_in_bound = np.logical_and(np.logical_and(x_in_bound, y_in_bound), z_in_bound)
    pc = pc[xyz_in_bound]
    return pc
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--us_dir', type=str, default='./data/images/us')
    parser.add_argument('--seg_dir', type=str, default='./data/images/us_seg')
    parser.add_argument('--probe_pose', type=str, default='./data/probe_pose/pose_optimize.npz')
    parser.add_argument('--img_to_probe', type=str, default='./data/img_to_probe/img_to_probe.npz')
    args = parser.parse_args()

    dataset = Dataset(args.us_dir, args.seg_dir, args.probe_pose, args.img_to_probe)

    grid_size = 0.5 # [mm]
    max_bound = 200 # [mm]
    n_grid = int(max_bound / grid_size)
    voxels = np.zeros((n_grid, n_grid, n_grid), np.float32)
    count = np.zeros((n_grid, n_grid, n_grid), np.int32)

    center = None

    point_clouds = []
    brightness_thres = 0.1

    for i in tqdm(range(len(dataset))):
        us_img, seg, R_us_to_cam, t_us_to_cam, scale_xy = dataset[i]

        cam_pc = img_to_cam_pc(seg[0], R_us_to_cam, t_us_to_cam, scale_xy)

        if center is None:
            center = np.mean(cam_pc[:, :3], axis=0)

        cam_pc = center_and_reduce(cam_pc, center, max_bound, brightness_thres)
        if len(cam_pc) == 0:
            continue
        xyz = cam_pc[:, :3]
        color = np.tile(cam_pc[:, 3:], (1, 3))

        v_sampled, _, c_sampled = pcu.downsample_point_cloud_voxel_grid(
                grid_size, np.ascontiguousarray(xyz), None, np.ascontiguousarray(color), 
                (-max_bound/2, -max_bound/2, -max_bound/2), 
                (max_bound/2, max_bound/2, max_bound/2))
        v_sampled = np.atleast_2d(v_sampled)
        c_sampled = np.atleast_2d(c_sampled)
        
        v_sampled = (v_sampled + max_bound / 2) / grid_size
        v_sampled = v_sampled.astype(np.int32)
        voxels[v_sampled[:, 0], v_sampled[:, 1], v_sampled[:, 2]] += c_sampled[:, 0]
        count[v_sampled[:, 0], v_sampled[:, 1], v_sampled[:, 2]] += 1

    voxels = gputools.gaussian_filter(voxels / (count + 1e-9), 3, 2)

    print('Rendering results')
    proj = volume_render(voxels, 'liver', bg=(255, 255, 255), size=(400, 400))

    for i in range(proj.shape[0]):
        cv2.imshow('img', proj[i])
        cv2.imwrite(str(i).zfill(6)+'.png', proj[i])
        cv2.waitKey(100)