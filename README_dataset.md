
```bash

cd VHAP && conda activate gaussian-avatars

gpustat
export CUDA_VISIBLE_DEVICES=1
# 1. Preprocess
conda install -c conda-forge ffmpeg
git clone https://github.com/PeterL1n/RobustVideoMatting.git

python vhap/preprocess_video.py \
--input /home/zhangzhh12024/Avatars/data/mono/luo0526/luo0526.mp4 \
--matting_method robust_video_matting \
--downsample_scales 2 4  --target-fps 5

# 2. FLAME ->
SUBJECT="104"
SEQUENCE="EMO-1" 
python vhap/track_nersemble_v2.py --data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/nersemble_v2/${SUBJECT}_${SEQUENCE}" \
--data.subject $SUBJECT --data.sequence $SEQUENCE \
--data.no-use-color-correction \
--data.n_downsample_rgb 4




SUBJECT="luo_7"
python vhap/track_nersemble_v2.py --data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/nersemble_v2/${SUBJECT}_EMO-1" \
--data.subject ${SUBJECT} --data.sequence "EMO-1" \

--data.no-use-color-correction  --optimize-cam


SUBJECT="luo_128"
SEQUENCE="EMO-1" 
time python vhap/track_nersemble_v2.py --data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/nersemble_v2/${SUBJECT}_EMO-1" \
--data.subject ${SUBJECT} --data.sequence "EMO-1" \
--data.no-use-color-correction --optimize-cam \
--data.image-size-during-calibration 2048 2048 \
--data.n_downsample_rgb 4


python vhap/export_as_nerf_dataset.py \
--src_folder "../output/nersemble_v2/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../output/export/${SUBJECT}_${SEQUENCE}" \
--background-color white













SUBJECT="luo_40"
SUBJECT="luo_40_mini"

python vhap/track_nersemble_v2.py --data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/nersemble_v2/${SUBJECT}_EMO-1" \
--data.subject ${SUBJECT} --data.sequence "EMO-1" \
--data.image-size-during-calibration 2048 2048 \
--data.no-use-color-correction  --optimize-cam --data.cam-to-img 



python vhap/export_as_nerf_dataset.py \
--src_folder "../output/nersemble_v2/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../export/nersemble_v2/${SUBJECT}_${SEQUENCE}_${DATE}" \
--background-color white

# 3. export_as_nerf_dataset

SUBJECT="luo_40_mini"
SEQUENCE="EMO-1" 
DATE=0526


python vhap/export_as_nerf_dataset_cam_to_img.py \
--src_folder "../output/nersemble_v2/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../export/nersemble_v2/${SUBJECT}_${SEQUENCE}_${DATE}" \
--background-color white

