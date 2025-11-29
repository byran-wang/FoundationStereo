# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import copy
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
import glob
import numpy as np

def save_depth(depth, filename):
    depth = np.nan_to_num(depth, nan=0.0)
    depth[depth > 8.0] = 0
    # Set depth to 0 if NaN
    depth_scaled = (depth / args.depth_scale).astype(np.uint16)
    depth_lsb = np.bitwise_and(depth_scaled, 0xFF)  # Least significant byte
    depth_msb = np.right_shift(depth_scaled, 8)  # Most significant byte
    depth_encoded = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    depth_encoded[..., 2] = depth_lsb
    depth_encoded[..., 1] = depth_msb    
    cv2.imwrite(filename, depth_encoded)

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_dir', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_dir', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--remove_black_pixels', default=1, type=int, help='remove black pixels in the right image, which are often invalid measurements')
  parser.add_argument('--denoise_cloud', action='store_true', help='whether to denoise the point cloud')
  parser.add_argument('--visualize_cloud', action='store_true', help='whether to visualize the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--ply_interval', type=int, default=1, help='interval to save point cloud')
  parser.add_argument('--ply_dir', type=str, default='ply', help='directory to save point cloud')
  parser.add_argument('--depth_scale', type=float, default=0.00012498664727900177, help='depth scale')
  parser.add_argument('--realsense', action='store_true', help='whether to use realsense images')
  parser.add_argument('--get_uncertainty', action='store_true', help='whether to compute and save depth uncertainty map')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)
  os.makedirs(args.ply_dir, exist_ok=True)
  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  file_extension = 'png'
  if args.realsense:
    file_extension = 'jpg'
  
  left_files = sorted(glob.glob(os.path.join(args.left_dir, f'*_left.{file_extension}')))
  right_files = sorted(glob.glob(os.path.join(args.right_dir, f'*_right.{file_extension}')))
  assert len(left_files) == len(right_files), "left and right files must have the same number of images"

  with open(args.intrinsic_file, 'rb') as f:
    data = pickle.load(f)
    K = np.array(data['stereo_camMat'])
    baseline = data['stereo_baseline']
  scale = args.scale
  K[:2] *= scale    

  for i, left_file in enumerate(left_files):
    left_file_index = left_file.split('/')[-1].split('.')[0].split('_left')[0]
    print(f"Processing... {left_file_index}")
    right_file = f"{args.right_dir}/{left_file_index}_right.{file_extension}"
    assert os.path.exists(right_file), f"right file {right_file} does not exist"

    if args.realsense:
      img0 = imageio.imread(left_file)
      img1 = imageio.imread(right_file)   
      # Convert back to 3-channel format for compatibility with the rest of the pipeline
      img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
      img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)   
    else:
      img0 = np.array(imageio.imread(left_file))
      img1 = np.array(imageio.imread(right_file))

      if img0.ndim==2:
        img0 = np.repeat(img0[...,None], 3, axis=2) # gray to 3-channel
      else:
        img0 = img0[:,:,:3]
      
      if img1.ndim==2:
        img1 = np.repeat(img1[...,None], 3, axis=2) # gray to 3-channel
      else:
        img1 = img1[:,:,:3]

    assert scale<=1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    import time
    start_time = time.time()
    with torch.cuda.amp.autocast(True):
      if not args.hiera:
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
      else:
        disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"forward time: {elapsed:.4f} seconds")
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis{left_file_index}.png', vis)
    logging.info(f"Output saved to {args.out_dir}")

    if args.remove_invisible:
      yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      us_right = xx-disp
      invalid = us_right<0
      disp[invalid] = np.inf
      if args.remove_black_pixels:
        us_right_int = np.rint(us_right).astype(int)
        in_bounds = (us_right_int >= 0) & (us_right_int < W)
        valid_mask = (~invalid) & in_bounds
        right_vals = np.zeros_like(disp, dtype=img1_gray.dtype)
        right_vals[valid_mask] = img1_gray[yy[valid_mask], us_right_int[valid_mask]]

      
      black_mask = valid_mask & (right_vals == 0)  # right pixel is black
      disp[black_mask] = np.inf



    depth = K[0,0]*baseline/disp
    save_depth(depth, f'{args.out_dir}/{left_file_index}.png')

    if args.get_uncertainty:
      # Calculate a depth uncertainty map using NCC around left/right matches
      patch_size = 5
      half_patch = patch_size // 2
      img0_gray_f = img0_gray.astype(np.float32)
      img1_gray_f = img1_gray.astype(np.float32)
      padded0 = np.pad(img0_gray_f, half_patch, mode='reflect')
      padded1 = np.pad(img1_gray_f, half_patch, mode='reflect')
      uncertainty = np.full((H, W), np.nan, dtype=np.float32)

      for y in range(H):
        y_pad = y + half_patch
        for x in range(W):
          d = disp[y, x]
          if not np.isfinite(d):
            continue
          x_r = x - d  # matching point in the right image
          x_r_int = int(round(x_r))
          if x_r_int < 0 or x_r_int >= W:
            continue

          x_pad = x + half_patch
          xr_pad = x_r_int + half_patch
          left_patch = padded0[y_pad-half_patch:y_pad+half_patch+1, x_pad-half_patch:x_pad+half_patch+1]
          right_patch = padded1[y_pad-half_patch:y_pad+half_patch+1, xr_pad-half_patch:xr_pad+half_patch+1]

          left_patch = left_patch - left_patch.mean()
          right_patch = right_patch - right_patch.mean()
          denom = np.sqrt((left_patch**2).sum() * (right_patch**2).sum()) + 1e-6
          ncc = float((left_patch * right_patch).sum() / denom)
          ncc = max(min(ncc, 1.0), -1.0)
          uncertainty[y, x] = 1.0 - ncc  # higher is less confident

      p90 = np.nanpercentile(uncertainty, 90) if np.isfinite(uncertainty).any() else 1.0
      uncertainty_to_save = np.nan_to_num(uncertainty, nan=p90)
      plt.imsave(f'{args.out_dir}/uncertainty{left_file_index}.png', uncertainty_to_save, cmap='viridis', vmax=p90)
    

    if args.get_pc and i % args.ply_interval == 0:

      xyz_map = depth2xyzmap(depth, K)
      pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
      keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
      keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
      pcd = pcd.select_by_index(keep_ids)
      o3d.io.write_point_cloud(f'{args.ply_dir}/{left_file_index}.ply', pcd)
      logging.info(f"PCL saved to {args.ply_dir}")

      if args.denoise_cloud:
        logging.info("denoise point cloud...")
        start_time = time.time()
        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"denoise time: {elapsed:.4f} seconds")
        inlier_cloud = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(f'{args.ply_dir}/cloud_denoise{left_file_index}.ply', inlier_cloud)
        pcd = inlier_cloud

      if args.visualize_cloud:
        logging.info("Visualizing point cloud. Press ESC to exit.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        vis.run()
        vis.destroy_window()
