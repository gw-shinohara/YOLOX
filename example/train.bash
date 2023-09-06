#/bin/bash -e
prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir
YOLOX_DATADIR="$prj_dir/datasets"

unlink "$YOLOX_DATADIR/2nd"
ln -s "/media/shinohara/SanDisk/grasped_data/2nd" "$YOLOX_DATADIR"

model="tiny_grasped"
project_name="Detect_white_cane_with_$model" #for wandb

python3 $prj_dir/tools/train.py \
-f  "$prj_dir/exps/custom/$model.py"\
-d 2 \
-b 64 \
-o \
--logger wandb wandb-project $project_name
