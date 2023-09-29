#/bi/bash

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir
exp_name="grasped_yolox_s"
exp_path="$prj_dir/exps/custom/$exp_name.py"

test_image_dir="/home/shinohara/Documents/YOLOX/datasets/white_cane_detection/2nd"
# test_image_path=$test_image_dir/2nd_stage/20190909/2nd_augmenttation_2m_3m_2person/val/1_confroom_background_87_526_072_538_009_0_0.jpg
# test_image_path="$test_image_dir/output_0519/val/knife_00005_23_0.jpg"
# test_image_path=$test_image_dir/20190909_output_right_3m/val/042_000_3_0.jpg
test_image_path=/home/shinohara/Documents/YOLOX/example/private/image_4.png

ckpt_name=best_ckpt
ckpt_file="/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_230929/$ckpt_name.pth"

python $prj_dir/tools/demo.py image -f $exp_path -c $ckpt_file --path $test_image_path --conf 0.1 --nms 0.45 --tsize 640 --save_result --device cpu
