#/bin/bash
prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir

test_image_dir="/home/shinohara/Documents/YOLOX/datasets/white_cane_detection/2nd"
# test_image_path=$test_image_dir/2nd_stage/20190909/2nd_augmenttation_2m_3m_2person/val/1_confroom_background_87_526_072_538_009_0_0.jpg
# test_image_path="$test_image_dir/output_0519/val/knife_00005_23_0.jpg"
test_image_path=$test_image_dir/20190909_output_right_3m/val/042_000_3_0.jpg
# test_image_path=/home/shinohara/Documents/YOLOX/example/private/image_4.png

output_image_dir="/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_230929/vis_res/onnx"
mkdir $output_image_dir

onnx_name=best_ckpt
onnx_file=/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_230929/onnx/$onnx_name.onnx
score_thr=0.1
device="cpu"

python3 $prj_dir/demo/ONNXRuntime/onnx_inference.py -m $onnx_file -i $test_image_path -o $output_image_dir -s $score_thr --input_shape 640,640

echo $output_image_dir
# code $output_image_dir