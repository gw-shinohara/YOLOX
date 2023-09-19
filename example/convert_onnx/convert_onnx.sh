#/bin/bash

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir

exp_name="grasped_yolox_s"
exp_path="$prj_dir/exps/custom/$exp_name.py"

weights_path="/home/shinohara/Documents/YOLOX/YOLOX_outputs/grasped_yolox_s_230919/grasped_yolox_s.pth"

#Note that to convert to openvino, opeset must be set to 10.
python3 $prj_dir/tools/export_onnx.py --output-name "$exp_name.onnx" -f $exp_path -c $weights_path --opset 10
