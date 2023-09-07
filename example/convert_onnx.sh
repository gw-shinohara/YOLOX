#/bin/bash

python3 -m pip install onnxruntime
prj_dir="/home/shinohara/Documents/YOLOX"

exp_name="tiny_grasped"
exp_path = "$prj_dir/exps/custom/$exp_name.py"

weights_path = "$prj_dir/YOLOX_outputs/$exp_name.pth"

cd $prj_dir
#Note that to convert to openvino, opeset must be set to 10.
python3 $prj_dir/tools/export_onnx.py --output-name $exp_name.onnx -f $exp_path -c $weights_path --opset 10
