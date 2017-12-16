import os
import tensorflow as tf
from PIL import Image
from feature import NPDFeature
from sklearn.model_selection import train_test_split

def import_data(name, convert_type='L', resize=None):
    """
    数据导入，判断为文件还是目录
    :param name: 输入的文件或者目录名
    :param convert_type: 需要转换的类型，默认为灰度
    :param resize: 希望重新赋予的size，默认为None
    :return: 如果是文件则返回Image对象，否则返回Image list，如果两者都不是则报错
    """
    if os.path.isfile(name):
        try:
            image = Image.open(name)
            image = image.convert(convert_type)
            if resize is not None:
                image = image.resize(resize)
            return image
        except Exception as e:
            print(e)
    elif os.path.isdir(name):
        file_list = [os.path.join(name, i) for i in os.listdir(name)]
        return [import_data(i, convert_type, resize) for i in file_list]
    else:
        raise FileExistsError("Error: wrong file/dir name can't be recognized!")


def split_data(data, label, train_size=0.8):
    """
    切分数据集为训练集和测试集
    :param data: 需要切分的数据集，list类型
    :param label: 对应的标签
    :param train_size: 切分的训练集大小，默认为0.8，默认测试集为0.2
    :return: 切分后的数据集
    """
    return train_test_split(data, label, train_size=train_size, test_size=1-train_size)  # 注意默认shuffle数据


if __name__ == "__main__":
    # write your code here
    # data reader
    face_data = import_data('datasets/original/face', resize=(24, 24))
    noface_data = import_data('datasets/original/nonface', resize=(24, 24))
    face_label = [0] * len(face_data)
    noface_label = [1] * len(noface_data)

    # data split
    # split_data(data, label)



    pass

