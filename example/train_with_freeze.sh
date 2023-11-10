#/bin/bash -e

python3 --version

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir
YOLOX_DATADIR="$prj_dir/datasets"
DATA_NAME="white_cane_detection"
DATA_VERTION="2nd"
unlink "$YOLOX_DATADIR/$DATA_NAME/$DATA_VERTION"
ln -s "/media/shinohara/SanDisk/grasped_data/$DATA_VERTION" "$YOLOX_DATADIR/$DATA_NAME"

exp_name="grasped_yolox_s_freeze"
exp_path="$prj_dir/exps/custom/$exp_name.py"
project_name="Detect_white_cane_with_$exp_name"
ckpt_path="/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_231018/best_ckpt.pth"

gnome-terminal -- tail -f $prj_dir/YOLOX_outputs/$exp_name/train_log.txt

python3 $prj_dir/tools/train.py \
-f $exp_path \
-d 1 \
-b 32 \
-c $ckpt_path
--cache
