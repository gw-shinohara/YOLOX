#/bin/bash -e

python3 --version

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir
YOLOX_DATADIR="$prj_dir/datasets"
DATA_NAME="finetune_tachikawa"
DATA_VERTION="tachikawa_station_20231101"
rm "$YOLOX_DATADIR/$DATA_NAME"
ln -s "/media/shinohara/SanDisk/$DATA_NAME" "$YOLOX_DATADIR/"

exp_name="grasped_yolox_s_tachikawa_finetune"
exp_path="$prj_dir/exps/custom/$exp_name.py"
project_name="Detect_white_cane_with_$exp_name"

gnome-terminal -- tail -f $prj_dir/YOLOX_outputs/$exp_name/train_log.txt

python3 $prj_dir/tools/train.py \
-f $exp_path \
-d 1 \
-b 32 \
--cache
