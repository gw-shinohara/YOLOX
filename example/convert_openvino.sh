#/bin/bash

# openvino
openvino_version = "openvino_2021"
setupvars_file = "/opt/intel/$openvino_2021/bin/setupvars.sh"
optimizer_dir = "path/to/your/toolkit_dir/$openvino_version/deployment_tools/model_optimizer"

# args
output_dir="$prj_dir/YOLOX_outputs"
weights_onnx_path = "$output_dir/$exp_name.onnx"
input_shapes = [1,3,640,640]

source $setupvars_file
cd $optimizer_dir
sudo $optimizer_dir/install_prerequisites/install_prerequisites_onnx.sh
python3 $optimizer_dir/mo.py --input_model $weights_onnx_path --input_shape $input_shapes --output_dir $output_dir
