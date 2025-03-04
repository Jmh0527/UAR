#!/bin/bash

# 设置路径
dataroot="/home/kh31/jingmh/AIM_test_feature"
checkpoint_dir="/home/kh31/IJCV/UAR/checkpoints/AIMClassifier_keepneg_L1"
script="eval.py"

# 循环从epoch_0到epoch_99
for epoch in {0..50}
do
  # 格式化 epoch 字符串为 epoch_0, epoch_1, ..., epoch_99
  epoch_str=$(printf "epoch_%d" $epoch)

  # 设置模型路径
  checkpoint_path="${checkpoint_dir}/${epoch_str}_model.pth"

  # 检查模型文件是否存在
  if [ -f "$checkpoint_path" ]; then
    
    # 运行 eval.py 脚本
    CUDA_VISIBLE_DEVICES=0 python $script --dataroot $dataroot --model AIMClassifier_keepneg_L1 --checkpoint $checkpoint_path --output /home/kh31/IJCV/linear_relu_keepneg_l1.txt
  else
    echo "模型文件 $checkpoint_path 不存在, 跳过..."
  fi
done
