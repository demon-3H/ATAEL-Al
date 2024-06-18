from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from medmnist import OrganAMNIST
from torch.utils.data import Dataset, DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler

def getCIFAR10DataLoader(root_path,batch_size,train="train"):
    if train == "train":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=False, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        return train_loader
    if train == "test":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        testset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return test_loader
class HAM10000Dataset(Dataset):
    def __init__(self, features, labels, root_dir, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.features.iloc[idx, 1] + '.jpg')
        image = Image.open(img_name)
        labels=self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

def getHAM10000DataLoader(root_path,batch_size,train="train"):
    # 加载数据
    data = pd.read_csv(root_path+'/HAM10000_metadata')
    label_mapping = {
        "nv": 0,
        "bkl": 1,
        "mel": 2,
        "bcc": 3,
        "akiec": 4,
        "vasc": 5,
        "df": 6
    }
    data['dx'] = data['dx'].map(label_mapping)

    # 划分特征和标签
    X = data.drop(['dx'], axis=1)
    y = data['dx']
    # 创建 RandomOverSampler 对象
    ros = RandomOverSampler(random_state=42)

    # 过采样训练数据
    X_resampled, y_resampled = ros.fit_resample(X, y)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    if train=="train":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = HAM10000Dataset(X_train, y_train, root_path+'/HAM10000', transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader
    if train=="test":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_dataset = HAM10000Dataset(X_test, y_test, root_path+'/HAM10000',transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader
def extract_data_labels(dataset):
        data_list = []
        labels_list = []
        for data, label in DataLoader(dataset, batch_size=1, shuffle=False):
            data_list.append(data.squeeze(0))  # 去掉批次维度
            labels_list.append(label.item())
        data_tensor = torch.stack(data_list)
        labels_tensor = torch.tensor(labels_list)
        X_np = data_tensor.numpy().reshape(len(data_tensor), -1)  # 将数据展平为二维数组
        y_np = labels_tensor.numpy()
        # 使用 RandomOverSampler 进行上采样
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_np, y_np)
        # 将上采样后的数据转换回 Tensor
        X_resampled = torch.tensor(X_resampled).reshape(-1, 1, 32, 32)
        y_resampled = torch.tensor(y_resampled)
        dataset_resampled = TensorDataset(X_resampled, y_resampled)
        return dataset_resampled

def getOrganAMNISTDataLoader(root_path,batch_size,train="train"):
    if train=="train":
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        trainset = OrganAMNIST(split='train', download=True, transform=transform_train)
        dataset_resampled = extract_data_labels(trainset)
        train_loader = torch.utils.data.DataLoader(dataset_resampled, batch_size=batch_size, shuffle=True)
        return train_loader
    if train=="test":
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        testset = OrganAMNIST(split='test', download=True, transform=transform_train)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return test_loader

def getDataLoader(data_name,root_path,batch_size,train="train"):
    if data_name=="CIFAR10":
        data_loader = getCIFAR10DataLoader(root_path, batch_size, train)
        return data_loader
    if data_name=="HAM10000":
        data_loader = getHAM10000DataLoader(root_path, batch_size, train)
        return data_loader
    if data_name=="OrganAMNIST":
        data_loader = getOrganAMNISTDataLoader(root_path, batch_size, train)
        return data_loader