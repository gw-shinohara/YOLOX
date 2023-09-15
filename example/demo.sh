#/bi/bash

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir

test_image_dir="/home/shinohara/Documents/YOLOX/datasets/white_cane_detection/2nd"
test_image_path="$test_image_dir/output_0519/val/whitestick_00077_81_0.jpg"

exp_name="grasped_yolox_s"
exp_path="$prj_dir/exps/custom/$exp_name.py"
project_name="Detect_white_cane_with_$exp_name"
ckpt_file=/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s/grasped_yolox_s_ckpt.pth

python $prj_dir/tools/demo.py image -f $exp_path -c $ckpt_file --path $test_image_path --conf 0.1 --nms 0.45 --tsize 640 --save_result --device gpu
code $test_image_path