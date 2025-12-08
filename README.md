# VHAP

**export**

the FLAME head is centered, moved to world origin (0,0,0)
camera is rearranged around.
a NeRF like dataset.

**alignment**

Align a custom mesh to FLAME canonical template using rigid Procrustes analysis.
Generates comprehensive outputs including meshes, landmarks, transformation data, and detailed reports.

```bash
cd VHAP

# Example: Zhang's MRI mesh
MESH_PATH=../data/MRI/MRI_zhang/MRI_zhang_skin_original.obj
LMK_PATH='../data/MRI/MRI_zhang/zhang_70lmks.npy'

#  
python ../Utils/mesh_align_to_flame.py \
--mesh-path ${MESH_PATH} --lmk-path ${LMK_PATH} \
--output-dir ../output/alignment \
--landmark-type static

```

```bash
python ../Utils/downsample_mesh.py \
-i ../data/MRI/MRI_zhang/MRI_zhang_skin_original.obj \
-o ../data/MRI/MRI_zhang \
-t 10000

python ../Utils/downsample_mesh.py \
-i ../output/alignment/meshes/user_mesh_aligned.obj \
-o ../output/alignment/meshes/ \
-t 10000



# 检测3D mesh上的landmarks (推荐使用53点版本)
python ../Utils/mesh_landmark_detector_v2.py  \
--input-mesh $MESH_PATH \
--output-path ../output/landmarks  



python ../Utils/mesh_landmark_detector.py  \
--input-mesh ../data/MRI/MRI_zhang/MRI_zhang_skin_original_downsampled.obj \
--output-path ../output/landmarks  --num_views 64


# 说明:
# - STAR检测器输出68个landmark点，可自动转换为53点(VHAP tracker兼容)
# - 重投影优化: 使用tracker思路迭代优化3D位置，提高精度
# - 渲染范围: 正面180度 (-90°到90°)，避免背面误检
# - 输出: landmarks_3d.npy, landmarks.pp, landmark_embedding.pkl, landmarks_vis.ply
# - 可视化: detections_2d/ (2D检测) + summary_visualization.png (综合对比)
# - 性能优化: KDTree加速mesh binding计算 (~100x)


```

## train

```bash



## require
SUBJECT="zhang_1107"
SEQUENCE="EMO-1" 

cd ~/Avatars/VHAP
conda activate gaussian-avatars
# python vhap/2_preprocess_frames.py \
# --sequence-dir "../data/nersemble_v2/${SUBJECT}/sequences/${SEQUENCE}/" \
# --input-frames-folder frames \
# --source-fps 50 \
# --target-fps 50 \
# --target-size 2048 \
# --downsample-scales 2 4 8

python vhap/3_rename_images.py "../data/nersemble_v2/${SUBJECT}"

```

```bash
# track
python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 3072 3072 \
--data.no-use-color-correction --optimize-cam  \
--data.n_downsample_rgb 4 --exp.no-photometric
# --exp.no-photometric --optimize-cam 
## sapiens

cd ~/Avatars/Utils/sapiens/lite/scripts/demo/torchscript
conda activate /home/zhangzhh12024/miniconda3/envs/sapiens_lite  
./seg.sh /home/zhangzhh12024/Avatars/data/nersemble_v2/${SUBJECT}/sequences/${SEQUENCE} 4
# takes ~1s/img

# export
python vhap/export_as_nerf_dataset.py \
--src_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../output/export/${SUBJECT}_${SEQUENCE}" \
--background-color white --n_downsample_rgb None

```

## cfg

```bash
# # track
# python vhap/track_nersemble_v2.py \
# --data.root_folder "../data/nersemble_v2" \
# --exp.output_folder "../output/tracking/${SUBJECT}_${SEQUENCE}" \
# --data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
# --data.image-size-during-calibration 2048 2048 \
# --data.no-use-color-correction --optimize-cam \
# --data.n_downsample_rgb 4
# # export
# python vhap/export_as_nerf_dataset.py \
# --src_folder "../output/tracking/${SUBJECT}_${SEQUENCE}" \
# --tgt_folder "../output/export/${SUBJECT}_${SEQUENCE}" \
# --background-color white --n_downsample_rgb None
```

```bash
SUBJECT="luo_128"
SEQUENCE="EMO-1" 

SUBJECT="zhang"
SEQUENCE="EMO-1" 


```bash
# SUBJECT="luo_128"
# SEQUENCE="rigid" 
# LMK_PATH='../data/MRI/70_luo.npy'
# MESH_PATH='../data/MRI/MRI_luotao_fit_scan_70lmk_+uv.obj'
SUBJECT="luo_128"
SEQUENCE="EMO-1" 
LMK_PATH='../data/MRI/full_landmark_points_53-70.npy'
MESH_PATH='../data/MRI/MRI_luotao_fit_scan_53lmk_+uv.obj'

SUBJECT="zhang"
SEQUENCE="EMO-1" 
MESH_PATH='../data/MRI/MRI_zhang/FLAME_fitting_zhang_70lmks.obj'
LMK_PATH='../data/MRI/MRI_zhang/zhang_70lmks.npy'

SUBJECT="XUE"export
SEQUENCE="EMO-1" 
MESH_PATH='/home/zhangzhh12024/Avatars/data/nersemble_v2/XUE/xueyiheng.nii.gz.obj'
LMK_PATH='/home/zhangzhh12024/Avatars/data/nersemble_v2/XUE/XUE.npy'
# track
python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 2048 2048 \
--data.no-use-color-correction --optimize-cam \
--data.n_downsample_rgb 4 \
--rigid_fitting --mesh-path ${MESH_PATH} --lmk-path ${LMK_PATH}

# export
python vhap/export_as_nerf_dataset.py \
--src_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../output/export/${SUBJECT}_${SEQUENCE}" \
--background-color white --n_downsample_rgb None

```

```bash
scp -P 22112 -r zhangzhh12024@10.15.49.36:"${REMOTE/#\/home/\/grpczm}"

# # track
# SUBJECT="luo_128"
# SEQUENCE="EMO-1" 

# python vhap/track_nersemble_v2.py \
# --data.root_folder "../data/nersemble_v2" \
# --exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
# --data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
# --data.image-size-during-calibration 2048 2048 \
# --data.no-use-color-correction --optimize-cam \
# --data.n_downsample_rgb 4 


```

### flame-fitting

## git

```bash
git checkout fedora

git push orgin fedora

git fetch origin

git reset --hard origin/main

git push origin fedora 
```

## rendering

```bash
conda activate gaussian-avatars

SCAN_PATH="/home/zhangzhh12024/Datasets/BMEdata/Chinese_Faces/九院_中国人颜面/04科技部课题北大/2023.06.10/151刘涵予/面扫"
SUBJECT=151刘涵予
SEQUENCE="EMO-1" 

python ../Utils/face_renderer.py $SCAN_PATH /home/zhangzhh12024/Avatars/data/nersemble_v2 \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--num_views=128 \
--image_size=2048 \
--view_range=-45,45 \
--elevation_range=-30,20 \
--distance_range=3,3.5 \
--fov_range=50,70 \
--mode random \
--mask \
--ambient_color=0.7,0.7,0.7 \
--diffuse_color=0.4,0.4,0.4 \
--specular_color=0.0,0.0,0.0 \
--data.downsample-scales 2 4 8 \
--output-name-format "cam_{i:06d}_000000.jpg"


SUBJECT=151刘涵予
SEQUENCE="EMO-1" 

MESH_PATH="/home/zhangzhh12024/Datasets/BMEdata/Chinese_Faces/九院_中国人颜面/04科技部课题北大/2023.06.10/151刘涵予/面扫/headscan.obj"
LMK_PATH="/home/zhangzhh12024/Avatars/data/nersemble_v2/151刘涵予/headscan_picked_points.npy"
# track
python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 2048 2048 \
--data.no-use-color-correction --optimize-cam \
--data.n_downsample_rgb 4 \
--rigid_fitting --mesh-path ${MESH_PATH} --lmk-path ${LMK_PATH}

# export
python vhap/export_as_nerf_dataset.py \
--src_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../output/export/${SUBJECT}_${SEQUENCE}" \
--background-color white --n_downsample_rgb None
```

```bash
conda activate vggt 
SCENE_PATH="/home/zhangzhh12024/Avatars/data/nersemble_v2/lhytest"
MESH_PATH="/home/zhangzhh12024/Datasets/BMEdata/Chinese_Faces/九院_中国人颜面/04科技部课题北大/2023.06.10/151刘涵予/面扫/headscan.obj"
# 基础mesh配准
python Utils/demo_mesh.py --scene_dir $SCENE_PATH --reference_mesh $MESH_PATH --output_registered_mesh --initial_mesh_scale 0.001

# 人脸重建 + BA + mesh配准
python Utils/demo_mesh.py --scene_dir $SCENE_PATH  --reference_mesh $MESH_PATH --use_ba --output_registered_mesh
```
