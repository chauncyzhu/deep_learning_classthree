import os
import pickle
import numpy as np
from PIL import Image
from itertools import chain
from feature import NPDFeature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ensemble import AdaBoostClassifier


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
            image = np.asarray(image)  # image change to array
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
    # index = range(len(data))
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=train_size, test_size=1-train_size)
    # x_train = [data[i] for i in x_train]
    # x_test = [data[i] for i in x_test]
    return  x_train, x_test, y_train, y_test # 注意默认shuffle数据


def prepare_data():
    # parameters
    dump_file = open('datasets/features/extract_features.pkl', 'wb')
    # data reader
    print("read image data...")
    face_data = import_data('datasets/original/face', resize=(24, 24))
    nonface_data = import_data('datasets/original/nonface', resize=(24, 24))
    face_label = [0] * len(face_data)
    noface_label = [1] * len(nonface_data)
    data = list(chain(*[face_data, nonface_data]))
    label = list(chain(*[face_label, noface_label]))

    # extract feature
    print("extract feature from image array...")
    data = [NPDFeature(i).extract() for i in data]

    # dump data
    print("dump data...")
    pickle.dump([data, label], dump_file)


if __name__ == '__main__':
    # prepare data
    # prepare_data()

    # read data from dump file
    print("read dump data...")
    dump_file = open('datasets/features/extract_features.pkl', 'rb')
    data, label = pickle.load(dump_file)

    # split data into train and test data
    print("split image data...")
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.7, test_size=0.3)

    # use adaboost classifier
    print("train adaboost classifier...")
    weak_classifier = DecisionTreeClassifier()
    n_weakers_limit = 10
    clf = AdaBoostClassifier(weak_classifier, n_weakers_limit)
    clf.fit(x_train, y_train)

    # train and predict
    clf.predict(x_test)

