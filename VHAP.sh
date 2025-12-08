#!/bin/sh
SUBJECT=zhang_1111
SEQUENCE=EMO-1


cd /home/zhangzhh12024/Avatars/VHAP
conda activate gaussian-avatars
python vhap/3_rename_images.py "../data/nersemble_v2/${SUBJECT}"


# track
python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 3072 3072 \
--data.no-use-color-correction --optimize-cam  \
--data.n_downsample_rgb 4 --exp.no-photometric


# sapiens
cd ~/Avatars/Utils/sapiens/lite/scripts/demo/torchscript
conda activate sapiens_lite  
./seg.sh /home/zhangzhh12024/Avatars/data/nersemble_v2/${SUBJECT}/sequences/${SEQUENCE} 5

# export
python vhap/export_as_nerf_dataset.py \
--src_folder "../output/jiuyuan/track/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../output/jiuyuan/export/${SUBJECT}_${SEQUENCE}" \
--background-color white --n_downsample_rgb None

# # # # # # # # # # # # 

SUBJECT="zhang_1111"
SEQUENCE="EMO-1" 
MESH_PATH='../data/MRI/MRI_zhang/FLAME_fitting_zhang_70lmks.obj'
LMK_PATH='../data/MRI/MRI_zhang/zhang_70lmks.npy'

# track
python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 3072 3072 \
--data.no-use-color-correction --optimize-cam \
--data.n_downsample_rgb 4 --exp.no-photometric \
--rigid_fitting --mesh-path ${MESH_PATH} --lmk-path ${LMK_PATH}



# lmk

python ../Utils/mesh_landmark_detector_v2.py  \
--input-mesh ../data/MRI/MRI_zhang/MRI_zhang_skin_original_downsampled.obj \
--output-path ../output/landmarks



# align
cd VHAP
MESH_PATH=../data/MRI/MRI_zhang/MRI_zhang_skin_original_downsampled.obj
LMK_PATH=../output/landmarks/landmarks.pp

#  
python ../Utils/mesh_align_to_flame.py \
--mesh-path ${MESH_PATH} --lmk-path ${LMK_PATH} \
--output-dir ../output/alignment \
--landmark-type static