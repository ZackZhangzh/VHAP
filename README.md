# VHAP: Versatile Head Alignment with Adaptive Appearance Priors

```bash



SUBJECT=P2_EMO-1
SEQUENCE=EMO-1
OUTPUT_FOLDER=/home/zhihao/Publications/output



python rename_images_camera.py \
--cfg.base-path /home/zhihao/Publications/data/nersemble_v2/$SUBJECT/sequences/$SEQUENCE \
--cfg.camera-mode sequential \
--cfg.image-mode simple_timestep

python vhap/track_nersemble_v2.py \
--data.root_folder "/home/zhihao/Publications/data/nersemble_v2" \
--data.subject $SUBJECT --data.sequence $SEQUENCE \
--exp.output_folder $OUTPUT_FOLDER/tracking/${SUBJECT}_${SEQUENCE} \
--data.image-size-during-calibration 3072 3072 \
--data.n_downsample_rgb 4 \
--data.no-use-color-correction \
--data.static_camera_motion




python vhap/export_as_nerf_dataset.py \
--src_folder $OUTPUT_FOLDER/tracking/${SUBJECT}_${SEQUENCE} \
--tgt_folder $OUTPUT_FOLDER/export/${SUBJECT}_${SEQUENCE} \
--background-color white 

```