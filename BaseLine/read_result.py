
import matplotlib.pyplot as plt
import numpy

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="SL")
#模型名字
parser.add_argument('-mn', '--model_name', type=str, default='mobilenet', help='the name of model')
#模型名字
parser.add_argument('-dn', '--data_name', type=str, default='OrganAMNIST', help='the name of data')

args = parser.parse_args()
def read_list_from_txt(filename):
    with open(filename, 'r') as file:
        lst = [line.strip() for line in file]
    return lst
def plt_model_accuracy(model_name):
    epochs=[i+1 for i in range(100)]
    # test_accuracy0 = read_list_from_txt(f"./result/{model_name}-0.txt")
    # test_accuracy0 = [round(float(num), 4) for num in test_accuracy0]
    test_accuracy1 = read_list_from_txt("./result/"+args.data_name+f"/{model_name}-1.txt")
    test_accuracy1 = [round(float(num), 4) for num in test_accuracy1]
    # test_accuracy2 = read_list_from_txt(f"./result/{model_name}-2.txt")
    # test_accuracy2 = [round(float(num), 4) for num in test_accuracy2]
    # print(max(test_accuracy0))
    print(max(test_accuracy1))
    # print(max(test_accuracy2))

    # test_accuracy2 = read_list_from_txt(f"./result/{model_name}_plus.txt")
    # test_accuracy2 = [round(float(num), 4) for num in test_accuracy2]
    # print(max(test_accuracy2))
    #
    #
    # test_accuracy3 = read_list_from_txt(f"./result/{model_name}_plus2.txt")
    # test_accuracy3 = [round(float(num), 4) for num in test_accuracy3]
    # print(max(test_accuracy3))


    # plt.plot(epochs, test_accuracy0,'b-', label=f'{model_name} Accuracy')
    plt.plot(epochs, test_accuracy1,'g-', label=f'{model_name}_ATAEL Accuracy')
    # plt.plot(epochs, test_accuracy2,'r-', label=f'{model_name}_ATAEL_AL Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # 设置y轴刻度从0开始
    plt.ylim(0, 1)
    plt.yticks(numpy.arange(0,1,0.1))
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{model_name}_accuracy.png')
    plt.show()
# plt_model_accuracy("VGG16")
plt_model_accuracy("ResNet18")
# plt_model_accuracy(args.model_name)
