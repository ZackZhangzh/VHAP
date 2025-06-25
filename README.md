#

## VHAP

```bash

conda activate gaussian-avatars
export CUDA_VISIBLE_DEVICES=1
```

```bash
cd VHAP 

# track
SUBJECT="luo_32"
SEQUENCE="EMO-1" 

python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 2048 2048 \
--data.no-use-color-correction --optimize-cam \
--data.n_downsample_rgb 4 


SUBJECT="luo_128"
SEQUENCE="EMO-1" 
LMK_PATH='../data/MRI/70_luo.npy'

python vhap/track_nersemble_v2.py \
--data.root_folder "../data/nersemble_v2" \
--exp.output_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--data.subject ${SUBJECT} --data.sequence ${SEQUENCE} \
--data.image-size-during-calibration 2048 2048 \
--data.no-use-color-correction --optimize-cam \
--data.n_downsample_rgb 4  --exp.no-photometric \
--lmk-path ${LMK_PATH}

# export
python vhap/export_as_nerf_dataset.py \
--src_folder "../output/track/${SUBJECT}_${SEQUENCE}" \
--tgt_folder "../output/export/${SUBJECT}_${SEQUENCE}" \
--background-color white --n_downsample_rgb None
```

### flame-fitting
