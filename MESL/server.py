import socket
import struct
import pickle
import torch.optim as optim
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import argparse
import datetime
import os
import sys
sys.path.append(os.path.abspath(".."))
import utils as utils
import  models
from torchdistill.common import  yaml_util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="SL")
#模型名字
parser.add_argument('-mn', '--model_name', type=str, default='mobilenet', help='the name of model')
#模型名字
parser.add_argument('-dn', '--data_name', type=str, default='OrganAMNIST', help='the name of data')
#gpu
parser.add_argument('-d', '--device', type=str, default='cuda:0', help='gpu id to use(e.g. 0,1,2,3)')
#主机ip
parser.add_argument('-ht', '--host', type=str, default='127.0.0.1', help='ip of host')
#端口号
parser.add_argument('-pt', '--port', type=int, default=10082, help='port of host')
# 阈值
parser.add_argument('-ut', "--update_threshold", type=float, default=0.0, help="update_threshold")

args = parser.parse_args()
config = yaml_util.load_yaml_file(os.path.expanduser("../config/"+args.data_name+"_config.yml"))

model_level_server_name=args.model_name+"-MESL-"+"server"

logger = utils.PrintLogger(args.data_name,model_level_server_name)


#设置GPU和CPU
device = args.device if torch.cuda.is_available() else "cpu"

#设置随机种子
utils.seed_everything()

client_net = models.ClientNet(args.model_name,config['inchannel']).to(device)
server_net = models.ServerNet(args.model_name,config['num_classes']).to(device)


if args.model_name == "mobilenet":
    decode = models.Decode(32)
    decode.load_state_dict(torch.load("./convdecoder1.pth"))
else:
    decode = models.Decode(64)
    decode.load_state_dict(torch.load("./convdecoder.pth"))
decode.eval()
decode.to(device)

#设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(server_net.parameters(),lr=config['learning_rate'],momentum=0.9,weight_decay=5e-4)


client_weights = copy.deepcopy(client_net.state_dict())
client_net.to(device)

total_sendsize_list = []
total_receivesize_list = []


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def train(conn):
    datasize = send_msg(conn, config['epochs'])  # send epoch

    total_batch, datasize = recv_msg(conn)  # get total_batch of train dataset


    logger.log("train start!")

    # 训练
    for e in range( config['epochs']):


        update = 0

        processBar = tqdm(range(total_batch))

        epoch_receivesize = 0
        epoch_sendsize = 0

        epoch_start_time = datetime.datetime.now()

        for i in processBar:

            optimizer.zero_grad()  # initialize all gradients to zero

            msg, datasize = recv_msg(conn)  # receive client message from socket

            epoch_receivesize += datasize
            total_receivesize_list.append(datasize)
            client_output_cpu = msg['client_output']  # client output tensor
            with torch.no_grad():
                client_output_cpu = decode(client_output_cpu.to(device))
            client_output = client_output_cpu.requires_grad_(True).to(device)
            label = msg['label']  # label
            label = label.clone().detach().long().to(device)
            output = server_net(client_output)  # forward propagation
            loss = criterion(output, label)  # calculates cross-entropy loss
            predictions = torch.argmax(output, dim=1)

            accuracy = torch.sum(predictions == label) / label.shape[0]
            loss.backward()  # backward propagation
            client_grad =client_output_cpu.grad.clone().detach()
            if loss.item() > args.update_threshold:
                update += 1
            else:
                client_grad = "abort"
            msg = {"grad_client": client_grad
                   }
            datasize = send_msg(conn, msg)
            epoch_sendsize += datasize
            total_sendsize_list.append(datasize)
            optimizer.step()

            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                       (e, config['epochs'], loss.item(), accuracy.item()))

        epoch_end_time = datetime.datetime.now()
        hour, minute, second = utils.time_regard(epoch_start_time, epoch_end_time)

        logger.log(
            "[%d/%d] Loss: %.4f, Acc: %.4f, sendsize: %dMB, receivesize: %dMB, totalsize: %dMB, time: %dh-%dm-%ds" %
            (e, config['epochs'], loss.item(), accuracy.item(), epoch_sendsize / pow(2, 20), epoch_receivesize / pow(2, 20),
             (epoch_sendsize + epoch_receivesize) / pow(2, 20), hour, minute, second))

        utils.save_model(server_net, model_level_server_name,args.data_name, e)





if __name__ == "__main__":
    # 监听连接
    s = socket.socket()
    s.bind((args.host, args.port))
    s.listen()
    conn, addr = s.accept()

    starttime = datetime.datetime.now()  # store start time

    train(conn)

    endtime = datetime.datetime.now()  # store end time

    hour, minute, second = utils.time_regard(starttime, endtime)
    logger.log("total time: %dh-%dm-%ds\n" % (hour, minute, second))

    logger.log('---total_sendsize_list---')
    logger.log("total_sendsize size: {} bytes".format(sum(total_sendsize_list)))
    logger.log('---total_receivesize_list---')
    logger.log("total receive sizes: {} bytes".format(sum(total_receivesize_list)))




