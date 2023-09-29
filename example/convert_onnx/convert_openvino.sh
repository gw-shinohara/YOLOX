#/bin/bash
prj_dir=/home/shinohara/Documents/YOLOX
# args
weights_dir=$prj_dir/YOLOX_outputs/grasped_yolox_s/onnx
output_dir="$prj_dir/YOLOX_outputs/grasped_yolox_s/openvino"

input_shapes="[1,3,640,640]"
optimizer_dir="/opt/intel/openvino_2021/deployment_tools/model_optimizer"
source $setupvars_file

model_name="best_ckpt"

# for model_name in best_ckpt epoch_10_ckpt epoch_11_ckpt epoch_12_ckpt epoch_13_ckpt epoch_14_ckpt epoch_15_ckpt epoch_1_ckpt epoch_2_ckpt epoch_3_ckpt epoch_4_ckpt epoch_5_ckpt epoch_6_ckpt epoch_7_ckpt epoch_8_ckpt epoch_9_ckpt last_epoch_ckpt latest_ckpt
# {
# weights_path="$weights_dir/$model_name.onnx"
# output_model_dir="$output_dir/$model_name"
# mkdir $output_model_dir
# echo $output_model_dir
python3 $optimizer_dir/mo.py --input_model $weights_path --input_shape $input_shapes --output_dir $output_model_dir
# }
