# document-classification-pytorch
The implementation of Hierarchical Attention Networks for Document Classification

requirements:
  pytorch
  pyltp
  jieba

The documents and sentences can be any length, I do not cut them off.  

  
train:
  
  The format of train and val data is csv, each row is a sample.
  The format is as follows:
      label '\t' token1 token2 ... tokenN
  The label and text are string and split by '\t'.
  The text must be tokenized, and split by space.

  (训练和验证数据都是csv格式，每一行为一个样本，格式如下：样本的label和text用\t分隔。针对text，事先分句，用空格分隔。
  具体可见train_data中的csv文件。)
  
  In config.py, set the parameter mode to 'train',
  and run:
      python main.py
      
  (在config.py中，将参数mode设置为'train',
  
  执行：
      python main.py即可)


dev：

  In config.py, set the parameter mode to 'eval',
  and run:
      python main.py
  The dev file is a txt file in test_data, it can be a normal document.
  
  (在config.py中，将参数mode设置为'eval', 参考dev_data_file为制定的csv文件
  
  执行：
      python main.py即可)
      
Experiments:
    The train data has 40k and 4 classes. It achieves 91.83% accuracy. I have not adjusted parameters systematically.

(实验：
  训练数据共40K，分4类，每类10k左右，在验证集上的准确率在91.83%，由于时间有限，还未系统地调参。)
