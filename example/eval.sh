#/bi/bash

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir

exp_name="grasped_yolox_s"
exp_path="$prj_dir/exps/custom/$exp_name.py"
project_name="Detect_white_cane_with_$exp_name"
ckpt_file=/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_230915/grasped_yolox_s.pth

python3 $prj_dir/tools/eval.py \
-f  $exp_path \
-c $ckpt_file \
-d 1 \
-b 32 \
--conf 0.001 \
