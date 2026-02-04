import os
data_path = "/home/simba/Documents/dataset/HOT3D/P0003_c701bd11/processed/stereo/214-1_1201-2"
cmd = "cd /home/simba/Documents/project/FoundationStereo && " \
    f"/home/simba/anaconda3/envs/foundation_stereo/bin/python " \
    "scripts/run_video.py " \
    f"--left_dir {data_path} " \
    f"--right_dir {data_path} " \
    f"--intrinsic_file {data_path}/0000.pkl " \
    f"--ckpt_dir ./pretrained_models/model_best_bp2.pth " \
    f"--out_dir {data_path}/depth_fs/ " \
    f"--ply_dir {data_path}/ply_fs/ " \
    f"--ply_interval 10"            
    # --realsense \
    # --denoise_cloud \
    # --visualize_cloud \
print(cmd)
os.system(cmd)