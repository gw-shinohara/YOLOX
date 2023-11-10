#/bi/bash

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir
exp_name="grasped_yolox_s_with_rotate"
exp_path="$prj_dir/exps/custom/$exp_name.py"

test_image_dir="/home/shinohara/Documents/YOLOX/datasets/white_cane_detection/tests"
#test_image_path="$test_image_dir/whitecane_tachikawa.mp4"
#test_image_path="$test_image_dir/whitestick_take1.mp4"
test_image_path="$test_image_dir/whitestick_take2.mp4"

ckpt_name=best_ckpt
ckpt_file="/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_with_rotate_copy/$ckpt_name.pth"

python $prj_dir/tools/demo.py video -f $exp_path -c $ckpt_file --path $test_image_path --conf 0.1 --nms 0.45 --tsize 640 --save_result --device "gpu"
