import os
import datetime
import struct
import pickle
import torch
import os
import numpy as np
import random
class PrintLogger:
    def __init__(self,dir_prefix, filename_prefix):
        self.filename_prefix = filename_prefix
        self.output_dir = 'logs/'+dir_prefix+'/'+filename_prefix
        self.output_file = self._get_output_file_name()

    def _get_output_file_name(self):
        """
        确定当天的输出文件名,如果同一天生成多个文件则自动添加序号。
        """
        file_count = 1
        today = datetime.date.today().strftime('%Y-%m-%d')
        base_filename = f"{file_count}_{self.filename_prefix}_{today}.txt"
        output_file = os.path.join(self.output_dir, base_filename)


        while os.path.exists(output_file):
            base_filename = f"{file_count}_{self.filename_prefix}_{today}.txt"
            output_file = os.path.join(self.output_dir, base_filename)
            file_count += 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        return output_file

    def log(self, *args, **kwargs):
        """
        将打印输出记录到指定文件,同时打印到控制台。
        """
        with open(self.output_file, 'a') as f:
            print(*args, file=f, **kwargs)
        print(*args, **kwargs)

    def log2(self, *args, **kwargs):
        """
        将打印输出记录到指定文件,同时打印到控制台。
        """
        with open(self.output_file, 'a') as f:
            print(*args, file=f, **kwargs)

def save_model(model, model_name,dir, training_batch):
    save_dir="./models/"+dir+"/"+model_name
    # 创建保存模型的文件夹（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名
    file_name = f'{model_name}_batch{training_batch}.pth'

    # 构建完整的文件路径
    file_path = os.path.join(save_dir, file_name)

    # 保存模型
    torch.save(model.state_dict(), file_path)

def time_regard(starttime,endtime):
    ti = (endtime - starttime).seconds
    hou = ti / 3600
    ti = ti % 3600
    sec = ti / 60
    ti = ti % 60
    return (hou, sec, ti)


def get_model_weight_files(model_dir):
    model_dir="./models/"+model_dir
    model_files = []
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.pth'):
            model_files.append(os.path.join(model_dir, file_name))
    return sorted(model_files, key=sort_by_batch)

def sort_by_batch(file_name):
    # 提取文件名中的批次数字
    batch_number = int(file_name.split('_batch')[1].split('.pth')[0])
    return batch_number


def model_test(model_files,model,test_loader,device):
    test_accuracy = []
    for j in range(len(model_files)):
        model.load_state_dict(torch.load(model_files[j]))
        model.eval()
        n = 0
        sum_acc = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predict = torch.max(outputs, 1)[1].data.squeeze()
            sum_acc += (predict == labels).sum().item() / labels.size(0)
            n += 1
        test_acc = sum_acc / n
        test_accuracy.append(test_acc)
        print('Epoch: %d, Test accuracy: %.4f ' % (j, test_acc))
    print("最大精度: ",max(test_accuracy))
    return test_accuracy

def model_test_sl(model_files1,model_files2,model1,model2,test_loader,device):
    test_accuracy = []
    for j in range(len(model_files1)):
        model1.load_state_dict(torch.load(model_files1[j]))
        # model1.eval()
        model2.load_state_dict(torch.load(model_files2[j]))
        # model2.eval()
        n = 0
        sum_acc = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            labels=labels.view(-1)
            outputs = model1(inputs)
            outputs = model2(outputs)
            predict = torch.max(outputs, 1)[1].data.squeeze()
            sum_acc += (predict == labels).sum().item() / labels.size(0)
            n += 1
        test_acc = sum_acc / n
        test_accuracy.append(test_acc)
        print('Epoch: %d, Test accuracy: %.4f ' % (j, test_acc))
    print("最大精度: ",max(test_accuracy))
    return test_accuracy


def seed_everything(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def save_list_to_txt(dir,filename,lst):
    save_dir = "./result/"+dir
    # 创建保存模型的文件夹（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 构建完整的文件路径
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

def compress(tensor):
    shape = tensor.shape
    tensor_reshaped = tensor.view(512,-1)
    non_zero_indices = torch.nonzero(tensor_reshaped)
    non_zero_values = tensor_reshaped[non_zero_indices[:, 0],non_zero_indices[:, 1]]
    # 组成三元组
    row = non_zero_indices[:, 0]
    int8_row = torch.as_tensor(row, dtype=torch.int16)
    bincount = torch.bincount(int8_row, minlength=512)
    row1 = non_zero_indices[:, 1]
    int8_row1 = torch.as_tensor(row1, dtype=torch.int16)
    value = non_zero_values
    # float16_value = value
    float16_value = torch.as_tensor(value, dtype=torch.float16)
    return bincount,int8_row1, float16_value,shape


def decompress(bincount,int8_row1,  float16_value,old_shape,device="cuda:0"):
    bincount = bincount.to(device)
    int8_row1 = int8_row1.to(device)
    float16_value = float16_value.to(device)
    unique_values = torch.arange(len(bincount)).to(device)
    tensor = torch.repeat_interleave(unique_values, bincount)
    bu_zore=512-tensor[-1]-1
    row = torch.stack([tensor, int8_row1], dim=1)
    # 获取索引tensor的形状,确定原始tensor的大小
    shape = torch.max(row, dim=0).values + 1
    # 创建全零tensor,并根据索引填充非零值
    tensor = torch.zeros(shape.tolist(), dtype=torch.float32).to(device)
    tensor[row[:, 0], row[:, 1]] = torch.as_tensor(float16_value, dtype=torch.float32)
    if bu_zore !=0:
        zero_tensor = torch.zeros(bu_zore, tensor.size()[1], dtype=torch.int64).to(device)

        # 使用 torch.cat() 函数将原始 tensor 和 zero_tensor 拼接在垂直方向上
        tensor = torch.cat([tensor, zero_tensor], dim=0)
    return tensor.view(tuple(old_shape))