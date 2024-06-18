#!/bin/bash

data_name=("OrganAMNIST" "HAM10000" "CIFAR10")

# 定义字符串数组
models=("VGG16" "ResNet18" "mobilenet")
# 使用for循环遍历数组
for data in "${data_name[@]}"
do
  for model in "${models[@]}"
    do
      echo "$data-$model"
      nohup python ./server.py -mn $model  -pt 10081 -l 1 -dn $data &
      sleep 2
      nohup python ./client.py -mn $model  -pt 10081 -l 1 -dn $data &
      wait $!
      sleep 10
      nohup python ./server.py -mn $model  -pt 10081 -l 2 -dn $data &
      sleep 2
      nohup python ./client.py -mn $model  -pt 10081 -l 2 -dn $data &
      wait $!
    done
done

