# -*-coding:utf-8-*-
import numpy as np
import cv2
import os
import pickle
import argparse


img_size = 112


def img_clip(img, outsize, img_path):
    '''
    首先截取图片中心180x180区域，然后缩放至img_size x img_size
    :param img:
    :param outsize:
    :param img_path:
    :return:
    '''
    hidden_size = 180
    h, w = img.shape
    if h < hidden_size or w < hidden_size:
        print(img_path)
        print('Image clip warnning! image size={}'.format(img.shape))
    else:
        h = (h-hidden_size) // 2
        w = (w-hidden_size) // 2
        img = img[h:h+hidden_size, w:w+hidden_size]
    return cv2.resize(img, (outsize, outsize), interpolation=cv2.INTER_CUBIC)

def get_data_info(data_dir):
    data_dirs = os.listdir(data_dir)
    data_info = {}
    for d in data_dirs:
        num = len(os.listdir(os.path.join(data_dir, d)))
        if num not in data_info:
            data_info[num] = [d]
        else:
            data_info[num].append(d)
    # sorted_keys = sorted(data_info.keys())
    # for key in sorted_keys:
    #     print(key, len(data_info[key]))
    return data_info

def get_asample(sample_path):
    img_paths = os.listdir(sample_path)
    img_paths = [int(i.split('.')[0]) for i in img_paths if i.split('.')[0].isdigit()]
    img_paths = ['{}.png'.format(i) for i in sorted(img_paths)]

    data = []
    for img_name in img_paths:
        img_path = os.path.join(sample_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('no this picture!', img_path)
            return None
        img = img.astype(np.float32)
        img = img_clip(img, img_size, img_path)
        # 归一化
        img -= np.mean(img)
        img /= np.std(img)

        data.append(img)
    return np.array(data)


def read_data(root_dir, data_info, id2word=None, word2label=None, test_data=False):
    data = []
    labels = []
    sorted_keys = sorted(data_info.keys(), reverse=True)
    for key in sorted_keys:
        # key: num of time step
        if key < 1:
            continue
        for s in data_info[key]:
            # s: 一个样本的文件名
            sample = get_asample(os.path.join(root_dir, s))
            if sample is not None:
                data.append(sample)
                if test_data:
                    labels.append(s)
                else:
                    labels.append(word2label[id2word[s]])
    return data, labels

def get_vocab(label_file):
    '''
    建立词表
    :param label_file: lip_train.txt文件的位置
    :return: 样本id与词语的对应id2word, 词语与下标的对应word2label
    '''
    id2word = {}
    word2label = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            ids, word = line.strip().split('\t')
            id2word[ids] = word
            if word not in word2label:
                word2label[word] = len(word2label)
    return id2word, word2label

def save_vocab(word2label, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in word2label.keys():
            f.write('{},{}\n'.format(word, word2label[word]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default='xin_data/train_dataset/lip_train/', type=str,
                        help='lip_train folder path')
    parser.add_argument("--test_path", default='xin_data/test_dataset/lip_test/', type=str,
                        help='lip_test folder path')
    parser.add_argument("--label_path", default='xin_data/train_dataset/lip_train.txt', type=str,
                        help='lip_train.txt file path')
    parser.add_argument("--save_path", default='data/', type=str,
                        help='the save path of the data')
    args = parser.parse_args()

    train_path = args.train_path
    test_path = args.test_path
    label_path = args.label_path
    save_path = args.save_path

    save_vocab_path = os.path.join(save_path, 'vocab.txt')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 解析并保存词表
    id2word, word2label = get_vocab(label_path)
    save_vocab(word2label, save_vocab_path)

    # 解析并保存训练数据
    data_info = get_data_info(train_path)
    data, labels = read_data(train_path, data_info, id2word, word2label)
    # for i in range(len(labels)):
    #     print(data[i].shape, labels[i])
    with open(os.path.join(save_path, 'train_data.dat'), 'wb') as f:
        pickle.dump(data, f)
        pickle.dump(labels, f)
    print('训练数据已保存.')

    # 解析并保存测试数据
    test_info = get_data_info(test_path)
    test_data, ids = read_data(test_path, test_info, test_data=True)
    with open(os.path.join(save_path, 'test_data.dat'), 'wb') as f:
        pickle.dump(test_data, f)
        pickle.dump(ids, f)
    print('测试数据已保存.')
