import struct
import socket
import pickle
import time
import torch
import torch.optim as optim
import argparse
import os
import sys
sys.path.append(os.path.abspath(".."))
import utils as utils
import getData as getData
import models
import torch.nn as nn
from torchdistill.common import  yaml_util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="MESL")

#模型名字
parser.add_argument('-mn', '--model_name', type=str, default='mobilenet', help='the name of model')
#模型名字
parser.add_argument('-dn', '--data_name', type=str, default='OrganAMNIST', help='the name of data')
#gpu
parser.add_argument('-d', '--device', type=str, default='cpu', help='gpu id to use(e.g. 0,1,2,3)')
#主机ip
parser.add_argument('-ht', '--host', type=str, default='127.0.0.1', help='ip of host')
#端口号
parser.add_argument('-pt', '--port', type=int, default=10082, help='port of host')


args = parser.parse_args()
config = yaml_util.load_yaml_file(os.path.expanduser("../config/"+args.data_name+"_config.yml"))

model_level_client_name=args.model_name+"-MESL-"+"client"

logger = utils.PrintLogger(args.data_name,model_level_client_name)

device = args.device if torch.cuda.is_available() else "cpu"

#设置随机种子
utils.seed_everything()

train_loader=getData.getDataLoader(args.data_name,config['path'],config['batch_size'])

total_batch = len(train_loader)
logger.log("训练集分批之后的组数：",total_batch)

#建立客户端模型
client_net = models.ClientNet(args.model_name,config['inchannel']).to(device)



if args.model_name == "mobilenet":
    encode = models.Encode(32)
    encode.load_state_dict(torch.load("./convencoder1.pth"))
else:
    encode = models.Encode(64)
    encode.load_state_dict(torch.load("./convencoder.pth"))
encode.eval()
encode.to(device)

#设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(client_net.parameters(),lr=config['learning_rate'],momentum=0.9,weight_decay=5e-4)


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def train():
    s = socket.socket()
    s.connect((args.host, args.port))
    epoch = recv_msg(s)

    msg = total_batch
    send_msg(s, msg)

    for e in range(epoch):

        for i, data in enumerate(train_loader):
            x, label = data
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = client_net(x)
            client_output = output.clone().detach()
            with torch.no_grad():
                client_output = encode(client_output)
            msg = {
                'client_output': client_output,
                'label': label
            }
            send_msg(s, msg)

            client_grad = recv_msg(s)["grad_client"]
            if client_grad=="abort":
                pass
            else:
                output.backward(client_grad.to(device))
                optimizer.step()

        utils.save_model(client_net, model_level_client_name,args.data_name, e)


if __name__ == "__main__":
    start_time = time.time()  # store start time
    logger.log("train start!")
    train()
    end_time = time.time()  # store end time
    logger.log("WorkingTime of ", device, ": {} sec".format(end_time - start_time))
