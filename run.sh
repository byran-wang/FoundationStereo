eval "$(conda shell.bash hook)"
conda activate foundation_stereo

# proxychains4 python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./test_outputs/

# object=bottles
# object=mirror
# object=wire
# object=mouse
# object=mouse_1
# object=mouse_hand
# object=drug_box
object=drug_box

# proxychains4 python scripts/run_demo.py \
#                 --left_file ./assets/${object}/left_image.png \
#                 --right_file ./assets/${object}/right_image.png \
#                 --intrinsic_file ./assets/K_ZED_HD.txt \
#                 --ckpt_dir ./pretrained_models/model_best_bp2.pth \
#                 --out_dir ./test_outputs/${object}/

# 



# dataset_dir=/home/simba/Documents/dataset/WonderHOI/realsense/
dataset_dir=/home/simba/Documents/dataset/WonderHOI/ZED_rotate/
# iterate over all scenes in dataset_dir
for scene in $(ls ${dataset_dir}); do
    echo "Processing scene: ${scene}"
    proxychains4 python scripts/run_video.py \
                --left_dir ${dataset_dir}/${scene}/ir/ \
                --right_dir ${dataset_dir}/${scene}/ir/ \
                --intrinsic_file ${dataset_dir}/${scene}/meta/0000.pkl \
                --ckpt_dir ./pretrained_models/model_best_bp2.pth \
                --out_dir ${dataset_dir}/${scene}/depth_fs/ \
                --ply_dir ${dataset_dir}/${scene}/ply_fs/ \
                --ply_interval 10 \
                
                # --realsense \
                # --denoise_cloud \
                # --visualize_cloud \

done
