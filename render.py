#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, rotation_to_quaternion, build_rotation, draw_cameras
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import imageio
import numpy as np
import einops


def quaternion_slerp(q1, q2, t):
    """
    Perform spherical linear interpolation (SLERP) between two quaternions.
    """
    dot_product = torch.clamp(torch.sum(q1 * q2, dim=-1), -1.0, 1.0)
    sign = torch.sign(dot_product)
    q1 = sign * q1
    dot_product = sign * dot_product
    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)
    q1_coeff = torch.sin((1 - t) * theta) / sin_theta
    q2_coeff = torch.sin(t * theta) / sin_theta
    return q1_coeff.unsqueeze(-1) * q1 + q2_coeff.unsqueeze(-1) * q2


def add_boarder(img, mode='interp', boarder_size=8):
    C, H, W = img.shape
    if mode == 'extrap':
        color = torch.tensor([1., 99/255, 71/255]).to(img)
    else:
        color = torch.tensor([0., 206/255, 209/255]).to(img)

    color = color.view(3, 1, 1).expand(C, boarder_size * 2 + H, boarder_size * 2 + W).clone()
    color[:, boarder_size:boarder_size + H, boarder_size:boarder_size + W] = img
    return color


def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))


def interpolate_time(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    if load2gpu_on_the_fly:
        view.load2device()
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering
    if load2gpu_on_the_fly:
        view.load2device()

    render_poses = torch.stack(render_wander_path(view, num_frames=frame), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )

    frame = 300
    # fallingball
    # render_poses = torch.stack([pose_spherical(angle, -5.0, 7.0)
    #                             for angle in np.linspace(20, 160, frame + 1)[:-1]], 0)
    # render_poses[:,:3,3] = render_poses[:,:3,3] + torch.Tensor([0, 0, 0.5]).to(render_poses)[None]
    # fan
    # render_poses = torch.stack([pose_spherical(angle, -15.0, 7.0)
    #                             for angle in np.linspace(-180, 180, frame + 1)[:-1]], 0)
    # whale
    # render_poses = torch.stack([pose_spherical(angle, -15.0, 7.0)
    #                             for angle in np.linspace(-90, 270, frame + 1)[:-1]], 0)
    # bat pendulums
    # render_poses = torch.stack([pose_spherical(angle, -15.0, 7.0)
    #                             for angle in np.linspace(0, 360, frame + 1)[:-1]], 0)
    # cloth robot
    render_poses = torch.stack([pose_spherical(angle, -15.0, 7.0)
                                for angle in np.linspace(-270, 90, frame + 1)[:-1]], 0)
    # spring
    # render_poses = torch.stack([pose_spherical(angle, -25.0, 7.0)
    #                             for angle in np.linspace(0, 180, frame + 1)[:-1]], 0)
    # # Chessboard
    # render_poses = torch.stack([pose_spherical(angle, -65.0, 4)
    #                             for angle in np.linspace(-90, 270, frame + 1)[:-1]], 0)
    # darkroom, Chessboard, dining, factory
    # render_poses = torch.stack([pose_spherical(angle, -45.0, 4)
    #                             for angle in np.linspace(-90, 270, frame + 1)[:-1]], 0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering
    if load2gpu_on_the_fly:
        view.load2device()

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        if i / (frame - 1) < 0.75:
            rendering = add_boarder(rendering, 'interp')
        else:
            rendering = add_boarder(rendering, 'extrap')
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )

    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]
    if load2gpu_on_the_fly:
        view.load2device()
        view_begin.load2device()
        view_end.load2device()

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view_original(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
                              deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )

    frame = 300
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    unique_views = []
    for view in views:
        is_duplicate = False
        if len(unique_views) > 0:
            for unique_view in unique_views:
                if np.allclose(view.R, unique_view.R) and np.allclose(view.T, unique_view.T):
                    is_duplicate = True
                    break
        if not is_duplicate:
            if load2gpu_on_the_fly:
                view.load2device()
                unique_views.append(view)
    print(f"In total ---------- {len(unique_views)} points")
    views = unique_views[::-3]
    view = views[0]
    if load2gpu_on_the_fly:
        view.load2device()
    renderings = []
    pcds = []
    T_curs = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx =  i * 1. / frame * (len(views) - 1)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        # if end_idx == len(views):
        #     break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        # R_begin = torch.tensor(view_begin.R)
        # q_begin = rotation_to_quaternion(R_begin[None])
        # R_end = torch.tensor(view_end.R)
        # q_end = rotation_to_quaternion(R_end[None])
        # t_begin = view_begin.T
        # t_end = view_end.T

        w2c_begin = np.eye(4)
        w2c_begin[:3, :3] = view_begin.R.T
        w2c_begin[:3, 3] = view_begin.T
        c2w_begin = np.linalg.inv(w2c_begin)
        Rw_begin, tw_begin = torch.tensor(c2w_begin[:3, :3]), c2w_begin[:3, 3]
        qw_begin = rotation_to_quaternion(Rw_begin[None])

        w2c_end = np.eye(4)
        w2c_end[:3, :3] = view_end.R.T
        w2c_end[:3, 3] = view_end.T
        c2w_end = np.linalg.inv(w2c_end)
        Rw_end, tw_end = torch.tensor(c2w_end[:3, :3]), c2w_end[:3, 3]
        qw_end = rotation_to_quaternion(Rw_end[None])

        ratio = query_idx - begin_idx

        # qw_cur = quaternion_slerp(qw_begin, qw_end, ratio)
        # Rw_cur = build_rotation(qw_cur, normalize=True).numpy()[0]
        Rw_cur = (1 - ratio) * Rw_begin + ratio * Rw_end
        tw_cur = (1 - ratio) * tw_begin + ratio * tw_end

        c2w_cur = np.eye(4)
        c2w_cur[:3, :3] = Rw_cur
        c2w_cur[:3, 3] = tw_cur
        w2c_cur = np.linalg.inv(c2w_cur)
        R_cur = w2c_cur[:3, :3].T
        T_cur = w2c_cur[:3, 3]
        T_curs.append(T_cur)

        view.reset_extrinsic(R_cur, T_cur)

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        if i / (frame - 1) < 0.75:
            rendering = add_boarder(rendering, 'interp')
        else:
            rendering = add_boarder(rendering, 'extrap')
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        pcds.append(draw_cameras(R_cur, T_cur, view.focal_x, view.image_height, view.image_width))

    print(f"In total ---------- {len(pcds)} points")
    # print(T_curs)
    # coord_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries(pcds)
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_val: bool, skip_test: bool,
                mode: str, static_threshold=0.01, fps=60):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train=skip_train, skip_val=skip_val, skip_test=skip_test)
        deform = DeformModel(max_time=dataset.max_time, light=dataset.light, physics_code=dataset.physics_code)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, static_threshold, fps=fps)

        if not skip_val:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "val", scene.loaded_iter,
                        scene.getValCameras(), gaussians, pipeline,
                        background, deform, static_threshold, fps=fps)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform, static_threshold, fps=fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--static_threshold", default=0.01, type=float)
    parser.add_argument('--fps', type=int, default=60)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.mode, args.static_threshold, args.fps)
