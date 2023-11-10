#/bin/bash

prj_dir="/home/shinohara/Documents/YOLOX"
cd $prj_dir

exp_name="grasped_yolox_s_with_rotate"
exp_path="$prj_dir/exps/custom/$exp_name.py"
output_dir="$prj_dir/YOLOX_outputs/$exp_name/onnx"
mkdir $output_dir

model_name=best_ckpt

#Note that to convert to openvino, opeset must be set to 10.

# for model_name in best_ckpt epoch_10_ckpt epoch_11_ckpt epoch_12_ckpt epoch_13_ckpt epoch_14_ckpt epoch_15_ckpt epoch_1_ckpt epoch_2_ckpt epoch_3_ckpt epoch_4_ckpt epoch_5_ckpt epoch_6_ckpt epoch_7_ckpt epoch_8_ckpt epoch_9_ckpt last_epoch_ckpt latest_ckpt
# { 
# echo $model_name
weights_path="$prj_dir/YOLOX_outputs/$exp_name/$model_name.pth"
output_path="$output_dir/$model_name.onnx"
echo "convert from :$weights_path"
echo "convert to   :$output_path"
python3 $prj_dir/tools/export_onnx.py -f $exp_path -c $weights_path --output-name $output_path --opset 10
# }

