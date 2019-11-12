# LipReading
2019年“创青春·交子杯”新网银行高校金融科技挑战赛-AI算法赛道唇语识别baseline

比赛网址:<https://www.dcjingsai.com/common/cmpt/2019%E5%B9%B4%E2%80%9C%E5%88%9B%E9%9D%92%E6%98%A5%C2%B7%E4%BA%A4%E5%AD%90%E6%9D%AF%E2%80%9D%E6%96%B0%E7%BD%91%E9%93%B6%E8%A1%8C%E9%AB%98%E6%A0%A1%E9%87%91%E8%9E%8D%E7%A7%91%E6%8A%80%E6%8C%91%E6%88%98%E8%B5%9B-AI%E7%AE%97%E6%B3%95%E8%B5%9B%E9%81%93_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html>  
基于论文“Combining Residual Networks with LSTMs for Lipreading”实现的唇语识别baseline
# 成绩
 单模单折未进行数据增强，线下Acc大约**0.53**, 线上成绩**0.56494**  
 ***如有帮助还请点个star***

# 环境需求
torch==1.2.0  
opencv-python==4.1.1.26

# 使用方法
### 1.处理数据
```shell
python data_process --train_path 新网银行唇语识别竞赛数据/1.训练集/lip_train/
                    --test_path 新网银行唇语识别竞赛数据/2.测试集/lip_test/
                    --label_path 新网银行唇语识别竞赛数据/1.训练集/lip_train.txt
                    --save_path data/
```
程序会读取并处理训练集和测试集数据，并在`data/`目录下缓存处理好的训练集文件`train_data.dat`、测试集文件`test_data.dat`以及词表`vocab.txt`

### 2.训练
```shell
python train.py --data_path data/train_data.dat
                --test_data_path data/test_data.dat
                --vocab_path data/vocab.txt
                --batch_size 16
                --epochs 40
```
程序会读取上一步处理的数据集和训练集文件，并根据`batch_size`填充数据，输入模型进行训练。完成训练后自动进行预测，并将预测结果保存为`submit.txt`

