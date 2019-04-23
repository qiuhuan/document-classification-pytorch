# document-classification-pytorch
The implementation of Hierarchical Attention Networks for Document Classification

requirements:
  pytorch
  pyltp
  jieba

  支持任意长度的文档和句子，没有做截断。
  
train:

  训练和验证数据都是csv格式，每一行为一个样本，格式如下：样本的label和text用\t分隔。针对text，事先分句，用空格分隔。
  具体可见train_data中的csv文件。
  
  在config.py中，将参数mode设置为'train',
  
  执行：
      python main.py即可


dev：

  在config.py中，将参数mode设置为'eval', 参考dev_data_file为制定的csv文件
  
  执行：
      python main.py即可
