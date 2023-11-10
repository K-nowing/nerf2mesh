import os
import cv2
import glob
import pickle
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, create_dodecahedron_cameras

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    return pose

def glossy2blender(_pose):
    P = np.array([[1,0,0],
                  [0,0,1],
                  [0,1,0]], dtype=np.float64)
    R = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,-1]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3,:3] = P @ _pose[:3,:3].transpose(1,0) @ R
    pose[:3,3:] = P @ _pose[:3, :3].transpose(1,0) @ (-_pose[:3, 3:])
    return pose



def visualize_poses(poses, size=0.1, bound=1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        if self.scale == -1:
            print(f'[WARN] --data_format nerf cannot auto-choose --scale, use 1 as default.')
            self.scale = 1
            
        self.training = self.type in ['train', 'all', 'trainval']

        if type =='all':
            pass
        elif type == 'trainval':
            pass
        else:
            pass
        self.img_num = len(glob.glob(f'{self.root_path}/*.pkl'))
        self.img_ids = [str(k) for k in range(self.img_num)]
        self.poses = np.zeros((self.img_num,4,4), dtype=np.float32)
        self.images = []
        self.H, self.W = None, None
        fl_x, fl_y, cx, cy = None, None, None, None
        for k in tqdm.tqdm(range(self.img_num), desc=f'Loading {type} data'):
            with open(f'{self.root_path}/{k}-camera.pkl', 'rb') as f:
                _pose, intrinsic = pickle.load(f)
                # glossy coordinate -> nerf blender coordinate
                pose = glossy2blender(_pose)
                
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                self.poses[k] = pose
                if fl_x is None:
                    fl_x = intrinsic[0,0]
                    fl_y = intrinsic[1,1]
                    cx = intrinsic[0,2]
                    cy = intrinsic[1,2]
                else:
                    assert fl_x == intrinsic[0,0]
                    assert fl_y == intrinsic[1,1]
                    assert cx == intrinsic[0,2]
                    assert cy == intrinsic[1,2]
            
            # remove background
            _image = cv2.imread(f'{self.root_path}/{k}.png')
            depth = cv2.imread(f'{self.root_path}/{k}-depth.png')[:,:,0]
            _image[depth==255] = 0
            h,w,_ = _image.shape
            image = np.zeros((h,w,4), np.uint8)
            image[...,:3] = _image
            image[...,3][depth<255] = 255
            
            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if self.downscale != 1:
                H, W, _ = image.shape
                image = cv2.resize(image, (self.W//self.downscale, self.H/self.downscale), interpolation=cv2.INTER_AREA)
            if self.H is None:
                self.H, self.W, _ = image.shape
            else:
                assert self.H == image.shape[0]
                assert self.W == image.shape[1]
                
            self.images.append(image)
        self.poses = torch.from_numpy(self.poses)
        self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
        
        # load intrinsics
        if self.downscale != 1:
            fl_x = fl_x / self.downscale
            fl_y = fl_y / self.downscale
            cx = cx/ self.downscale
            cy = cy/ self.downscale
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        
        #train_ids, test_ids = self.get_ids_split(self.img_ids, 'validation')
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        
        # [debug] uncomment to view all training poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses.numpy(), bound=self.opt.bound)

        # perspective projection matrix
        self.near = self.opt.min_near
        self.far = 1000 # infinite
        y = self.H / (2.0 * fl_y)
        aspect = self.W / self.H
        self.projection = np.array([[1/(y*aspect), 0, 0, 0], 
                                    [0, -1/y, 0, 0],
                                    [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                    [0, 0, -1, 0]], dtype=np.float32)

        self.projection = torch.from_numpy(self.projection)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(self.poses)
    
        # tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        # visualize_poses(dodecahedron_poses, bound=self.opt.bound, points=self.pts3d)
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps = self.projection.unsqueeze(0) @ torch.inverse(self.dodecahedron_poses)

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            self.projection = self.projection.to(self.device)
            self.mvps = self.mvps.to(self.device)


    def collate(self, index):

        B = len(index) # a list of length 1

        results = {'H': self.H, 'W': self.W}

        if self.training and self.opt.stage == 0:
            # randomly sample over images too
            num_rays = self.opt.num_rays

            if self.opt.random_image_batch:
                index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device)

        else:
            num_rays = -1

        poses = self.poses[index].to(self.device) # [N, 4, 4]

        rays = get_rays(poses, self.intrinsics, self.H, self.W, num_rays, self.opt.patch_size)

        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        results['index'] = index

        if self.opt.stage > 0:
            mvp = self.mvps[index].to(self.device)
            results['mvp'] = mvp

        if self.images is not None:
            
            if self.training and self.opt.stage == 0:
                images = self.images[index, rays['j'], rays['i']].float().to(self.device) / 255 # [N, 3/4]
            else:
                images = self.images[index].squeeze(0).float().to(self.device) / 255 # [H, W, 3/4]

            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)

            results['images'] = images
            
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
    
    @staticmethod
    def get_ids_split(img_ids, split_type='validation'):
        if split_type == 'validation':
            random.seed(6033)
            random.shuffle(img_ids)
            test_ids = img_ids[:1]
            train_ids = img_ids[1:]
        elif split_type=='test':
            with open('grossy_synthetic_split_128.pkl', 'rb') as f:
                test_ids, train_ids = pickle.load(f)
        else:
            raise NotImplementedError
        return train_ids, test_ids