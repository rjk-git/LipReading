# -*-coding:utf-8-*-
import numpy as np
import random
import torch
import os
import pickle
from LipModel import LipModel
from tqdm import tqdm
import argparse

def padding_batch(array_batch):
    data = []
    time_steps = [a.shape[0] for a in array_batch]
    max_timestpe = max(time_steps)
    for i, array in enumerate(array_batch):
        if array.shape[0] != max_timestpe:
            t, h, w = array.shape
            pad_arr = np.zeros((max_timestpe-t, h, w), dtype=np.float32)
            array_batch[i] = np.vstack((array, pad_arr))
        data.append(array_batch[i])
    return torch.tensor(data).unsqueeze(1)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_train_data(array_list, label_list, batch_size, test_data=False):
    train_data = []
    label_data = []
    num_data = len(array_list)
    num_batch = num_data // batch_size if num_data % batch_size == 0 else num_data // batch_size + 1

    batch_range = list(range(num_batch))
    random.shuffle(batch_range)
    bar = tqdm(batch_range)

    for i in bar:
        start = i * batch_size
        end = (i+1) * batch_size if (i+1) * batch_size < num_data else num_data

        train_data.append(padding_batch(array_list[start:end]))
        if test_data:
            label_data.append(label_list[start:end])
        else:
            label_data.append(torch.tensor(label_list[start:end]))
    bar.close()

    return train_data, label_data

def split_train_eval(array_list, label_list, num_eval):
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    eval_idx = random.sample(range(len(array_list)), num_eval)
    for i in range(len(array_list)):
        if i not in eval_idx:
            train_data.append(array_list[i])
            train_label.append(label_list[i])
        else:
            eval_data.append(array_list[i])
            eval_label.append(label_list[i])
    return train_data, train_label, eval_data, eval_label

def eval(model, eval_data, eval_label, device):
    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for step in range(len(eval_data)):
            batch_inputs = eval_data[step].to(device)
            batch_labels = eval_label[step].to(device)

            logist = model(batch_inputs)[0]
            count += logist.size(0)
            acc += torch.sum(torch.eq(torch.argmax(logist, dim=-1), batch_labels)).item()
    model.train()
    return acc/count

def predict(model, batch_size, model_path, data_path, vocab_path, result_to_save, device):
    load_chach = False
    ##############################
    #         模型加载
    ##############################
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    model.to(device)
    print('加载模型')

    id2label = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    ##############################
    #         数据加载
    ##############################
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
        test_ids = pickle.load(f)
    print('数据加载完成, data num = {}, label num = {}'.format(len(test_data), len(test_ids)))

    if load_chach:
        with open('test_catch.dat', 'rb') as f:
            test_data = pickle.load(f)
            test_ids = pickle.load(f)
        print('加载缓存数据')
    else:
        test_data, test_ids = get_train_data(test_data, test_ids, batch_size, test_data=True)
        with open('test_catch.dat', 'wb') as f:
            pickle.dump(test_data, f)
            pickle.dump(test_ids, f)
        print('缓存数据')
    print('pad填充完成, test batch num = {}'.format(len(test_data)))

    ##############################
    #            预测
    ##############################
    print('预测中...')
    pre_result = []
    with torch.no_grad():
        for step in range(len(test_data)):
            batch_inputs = test_data[step].to(device)
            logist = model(batch_inputs)[0]

            pred = torch.argmax(logist, dim=-1).tolist()
            assert len(pred) == len(test_ids[step])
            for i, ids in enumerate(test_ids[step]):
                pre_result.append(ids + ',' + id2label[pred[i]])
    with open(result_to_save, 'w', encoding='utf-8') as f:
        for line in pre_result:
            f.write(line + '\n')
    print('预测结果已保存至:', result_to_save)



def train(args):
    num_class = 313
    save_model = True

    data_path = args.data_path
    test_data_path = args.test_data_path
    vocab_path = args.vocab_path
    model_save_path = args.model_save_path
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    lr = args.lr
    log_step = args.log_step
    grad_clip = args.grad_clip
    num_eval = args.num_eval
    eval_batch = args.eval_batch
    load_cache = args.load_cache


    ##############################
    #         模型加载
    ##############################
    model = LipModel(1, num_class)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('加载模型')
    num_param = get_parameter_number(model)
    print('total parameter: {}, trainable parameter: {}'.format(num_param['Total'], num_param['Trainable']))

    ##############################
    #         数据加载
    ##############################
    with open(data_path, 'rb') as f:
        train_data = pickle.load(f)
        label_data = pickle.load(f)
    print('数据加载完成, data num = {}, label num = {}'.format(len(train_data), len(label_data)))

    ##############################
    #         数据处理
    ##############################
    train_data, label_data, eval_data, eval_label = split_train_eval(train_data, label_data, num_eval)
    print('数据分割完成, train data num = {}, eval data num = {}'.format(len(train_data), len(eval_data)))
    if load_cache:
        with open('catch.dat', 'rb') as f:
            train_data = pickle.load(f)
            label_data = pickle.load(f)
            eval_data = pickle.load(f)
            eval_label = pickle.load(f)
        print('加载缓存数据')
    else:
        train_data, label_data = get_train_data(train_data, label_data, batch_size)
        eval_data, eval_label = get_train_data(eval_data, eval_label, eval_batch)
        with open('cache.dat', 'wb') as f:
            pickle.dump(train_data, f)
            pickle.dump(label_data, f)
            pickle.dump(eval_data, f)
            pickle.dump(eval_label, f)
        print('缓存数据')
    print('pad填充完成, train batch num = {}, eval batch num = {}'.format(len(train_data), len(eval_data)))

    ##############################
    #            训练
    ##############################
    best_acc = -1
    pred_label = []
    true_label = []
    for epoch in range(1, epochs+1):
        avg_loss = 0
        data_indexs = list(range(len(train_data)))
        random.shuffle(data_indexs)
        for step, data_idx in enumerate(data_indexs):
            batch_inputs = train_data[data_idx].to(device)
            batch_labels = label_data[data_idx].to(device)

            logist, loss = model(batch_inputs, batch_labels)
            logist = torch.argmax(logist, dim=-1)
            loss = loss.mean()

            pred_label.append(logist)
            true_label.append(batch_labels)
            avg_loss += loss.item()
            if step % log_step == 0:
                pred_acc = torch.mean(torch.eq(torch.cat(pred_label), torch.cat(true_label)).float()).item()
                print('epoch={}, step={}, timestep={}, loss={:.3f}, pred acc={:.3f}'.format(
                    epoch, step, batch_inputs.size(2), avg_loss/(step+1), pred_acc))
                pred_label = []
                true_label = []

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
        acc = eval(model, eval_data, eval_label, device=device)
        print('='*100)
        print('epoch = {}, Avg train loss = {}, Acc = {}'.format(epoch, avg_loss/len(train_data), acc))

        if save_model and acc >= best_acc:
            model_to_save = model.module if hasattr(model, 'module') else model
            with open(model_save_path, 'wb') as f:
                torch.save(model_to_save.state_dict(), f)
            print('保存模型:', model_save_path)
            best_acc = acc
        print('=' * 100)
    print('训练完成: best acc =', best_acc)

    print('预测', '='*100)
    predict(model, eval_batch, model_save_path, test_data_path,
            vocab_path=vocab_path, result_to_save='submit.csv', device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='data/train_data.dat', type=str,
                        help='train data path')
    parser.add_argument("--test_data_path", default='data/test_data.dat', type=str,
                        help='test data path')
    parser.add_argument("--vocab_path", default='data/vocab.txt', type=str,
                        help='vocab path')
    parser.add_argument("--model_save_path", default='model/model.pt', type=str,
                        help='the path model to save')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--grad_clip", default=0.5, type=float)

    parser.add_argument("--log_step", default=100, type=int, help='print information interval')
    parser.add_argument("--num_eval", default=1000, type=int, help='number of verification set')
    parser.add_argument("--eval_batch", default=4, type=int, help='batch size of verify')
    parser.add_argument('--load_cache', action='store_true')
    args = parser.parse_args()

    train(args)
