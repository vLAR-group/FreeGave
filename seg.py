import torch
import torch.nn as nn
from scene import Scene, DeformModel, GaussianModel
import os
from os import makedirs
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import numpy as np
from fast_pytorch_kmeans import KMeans
import open3d as o3d
from utils.point_visual_util import build_pointcloud_segm
import einops

from utils.point_visual_util import pc_flow_to_sphere


np.random.seed(233)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def cluster_gaussians(dataset: ModelParams, iteration: int, n_keys=10, visualization=True, max_time=0.5, smooth=0.1, scene_type='unknown'):

    with (torch.no_grad()):
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train=True)
        deform = DeformModel(dataset.max_time, light=dataset.light, physics_code=dataset.physics_code)
        deform.load_weights(dataset.model_path)
        label_save_path = os.path.join(dataset.model_path, 'seg')
        makedirs(label_save_path, exist_ok=True)

        xyz = gaussians.get_xyz
        deform_code = deform.code_field(xyz)
        deform_seg = deform.code_field.seg(deform_code)

        dxyz = deform.deform.get_translation(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)

        # static mask
        sampled_time = torch.arange(0, int(100 * max_time), 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(int(100 * max_time)):
            dxyz_t = deform.deform.get_translation(xyz, sampled_time[:, i:i + 1], deform_code)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < 0.01)
        motion_mask = ~static_mask

        # feature for k-means
        feature = torch.cat([deform_seg, smooth * xyz], dim=-1) * motion_mask.unsqueeze(-1)

        kmeans = KMeans(n_clusters=n_keys, mode='euclidean', verbose=1)
        labels = kmeans.fit_predict(feature)

        xyz0 = xyz + dxyz
        if scene_type == 'others':
            surround = torch.tensor([[-2.5, -2.5, 0.05], [2.5, 2.5, 5.95]], device='cuda')
            outside = (xyz0 < surround[0]).any(dim=-1) | (xyz0 > surround[1]).any(dim=-1)
            most = labels[outside].bincount().argmax()
            labels[outside] = most
        elif scene_type == 'dining':
            surround = torch.tensor([[-2.5, -2.5, 0.64], [2.5, 2.5, 5.95]], device='cuda')
            outside = (xyz0 < surround[0]).any(dim=-1) | (xyz0 > surround[1]).any(dim=-1)
            most = labels[outside].bincount().argmax()
            labels[outside] = most

        np.save(f"{label_save_path}/labels.npy", einops.asnumpy(labels))

        # visualization
        prev_dxyz = dxyz
        flow_mesh = [] # o3d.geometry.TriangleMesh.create_sphere(radius=0.0001, resolution=resolution)
        mask = np.random.randint(motion_mask.sum().item(), size=700)
        prev_i = 0
        flow_label = einops.asnumpy(labels[motion_mask][mask])
        # for i in [0, 11, 22, 33, 44, 55, 66, 77, 88]:
        #     t = torch.ones_like(xyz[..., :1]) * i / 88
        for i in [0, 10, 20, 30, 40, 50, 59]:
            t = torch.ones_like(xyz[..., :1]) * i / 60
            d_xyz = torch.zeros_like(gaussians.get_xyz)
            d_rotation = torch.zeros_like(gaussians.get_rotation)
            d_scaling = torch.zeros_like(gaussians.get_scaling)

            d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
                xyz[motion_mask], t[motion_mask], deform_code[motion_mask]
            )
            if static_mask.sum() > 0:
                d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
                    xyz[static_mask], t[static_mask] * 0., deform_code[static_mask]
                )

            points = einops.asnumpy(xyz + d_xyz)
            pcds = o3d.geometry.PointCloud()
            pcds = pcds + build_pointcloud_segm(points, einops.asnumpy(labels))

            if i > 0:
                flow = einops.asnumpy((d_xyz - prev_dxyz)[motion_mask][mask])
                color = (flow / np.linalg.norm(flow, axis=-1, keepdims=True).mean(axis=0) / 2 + 0.5)
                color = color.clip(0, 1)
                start_pts = (xyz + d_xyz)[motion_mask][mask]
                for k in range(n_keys):
                    valid = flow_label == k
                    if valid.sum() == 0:
                        continue
                    cur_mesh = pc_flow_to_sphere(einops.asnumpy(start_pts)[valid], flow[valid], color=color[valid])
                    flow_mesh.append(cur_mesh)
                    # o3d.io.write_triangle_mesh(f"{save_path}/seg_{k:02d}/flow_{prev_i:03d}->{i:03d}.ply", cur_mesh, write_ascii=True)
                prev_dxyz = d_xyz
                prev_i = i

            # o3d.io.write_point_cloud(f"{save_path}/segmentations_{i:03d}.ply", pcds, write_ascii=True)

        # with open(f"{save_path}/static_mask.npy", 'wb') as f:
        #     np.save(f, einops.asnumpy(static_mask))
        #
        # with open(f"{save_path}/flow_label.npy", 'wb') as f:
        #     flow_label = labels[motion_mask]
        #     flow_label = einops.asnumpy(flow_label)[mask]
        #     np.save(f, flow_label)

        if visualization:
            o3d.visualization.draw_geometries([pcds] + flow_mesh)

            points = einops.asnumpy((xyz + dxyz)[static_mask])
            pcds = o3d.geometry.PointCloud()
            pcds = pcds + build_pointcloud_segm(points, einops.asnumpy(labels[static_mask]))
            o3d.visualization.draw_geometries([pcds])
            #
            points = einops.asnumpy((xyz + dxyz)[~static_mask])
            pcds = o3d.geometry.PointCloud()
            pcds = pcds + build_pointcloud_segm(points, einops.asnumpy(labels[~static_mask]))
            o3d.visualization.draw_geometries([pcds])


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--K", default=512, type=int)
    parser.add_argument("--smooth", default=0.1, type=float)
    parser.add_argument("--scene", default="unknown", type=str, help="Scene name for saving results")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    cluster_gaussians(model.extract(args), args.iteration, n_keys=args.K, visualization=args.vis, max_time=args.max_time, smooth=args.smooth, scene_type=args.scene)

