#/bin/bash
prj_dir=/home/shinohara/Documents/YOLOX
# args
exp_name="grasped_yolox_s"
weights_dir=$prj_dir/YOLOX_outputs/grasped_yolox_s_230919
output_dir="$weights_dir/openvino"
mkdir $output_dir
weights_onnx_path="$weights_dir/$exp_name.onnx"
input_shapes="[1,3,640,640]"
optimizer_dir="/opt/intel/openvino_2021/deployment_tools/model_optimizer"
source $setupvars_file
python3 $optimizer_dir/mo.py --input_model $weights_onnx_path --input_shape $input_shapes --output_dir $output_dir
