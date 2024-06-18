#!/bin/bash

#data_name=("OrganAMNIST" "HAM10000" "CIFAR10")
data_name=("OrganAMNIST")

# 定义字符串数组
models=("ResNet18" "mobilenet")
#models=("VGG16")

# 使用for循环遍历数组
for data in "${data_name[@]}"
do
  for model in "${models[@]}"
    do
      echo "$data-$model"
      nohup python ./server.py -mn $model  -pt 10082 -dn $data &
      sleep 2
      nohup python ./client.py -mn $model  -pt 10082 -dn $data &
      wait $!
      sleep 10
    done
done

