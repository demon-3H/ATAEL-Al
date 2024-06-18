import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import models
import os
import sys
sys.path.append(os.path.abspath(".."))
import utils as utils
import getData as getData
import argparse
from torchdistill.common import  yaml_util
from medmnist import OrganAMNIST
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="SL")
#模型名字
parser.add_argument('-mn', '--model_name', type=str, default='mobilenet', help='the name of model')
#模型等级
parser.add_argument('-l', '--level', type=int, default=0, help='the level of model')
#模型名字
parser.add_argument('-dn', '--data_name', type=str, default='OrganAMNIST', help='the name of data')

args = parser.parse_args()
config = yaml_util.load_yaml_file(os.path.expanduser("../config/"+args.data_name+"_config.yml"))

utils.seed_everything()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

test_loader=getData.getDataLoader(args.data_name,config['path'],config['batch_size'],"test")

# 获取模型文件列表
model_server_baseline_files = utils.get_model_weight_files(args.data_name+"/"+args.model_name+'-baseline-server')
print(model_server_baseline_files)
model_client_baseline_files = utils.get_model_weight_files(args.data_name+"/"+args.model_name+'-baseline-client')
print(model_client_baseline_files)

model_client_net = models.ClientNet(args.model_name,config['inchannel']).to(device)
model_server_net = models.ServerNet(args.model_name,config['num_classes']).to(device)

test_accuracy=utils.model_test_sl(model_client_baseline_files,model_server_baseline_files,model_client_net,model_server_net,test_loader,device)
utils.save_list_to_txt(args.data_name,args.model_name+'-'+str(args.level)+'.txt',test_accuracy)
