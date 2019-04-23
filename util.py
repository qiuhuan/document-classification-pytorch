from pyltp import SentenceSplitter
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
from config import vocabulary_file, labels_file
import jieba
import pdb

#the inference for data
def getEncodeDecode(txt,istoken):
    data_list = [w.strip() for w in open(txt,'r')]
    decode,encode = {},{}
    if istoken:
        data_list = ['<pad>','<unk>'] + data_list
    for i, w in enumerate(data_list):
        decode[i] = w
        encode[w] = i
    return encode,decode

token2idx,idx2token = getEncodeDecode(vocabulary_file,istoken = True)
label2idx,idx2label = getEncodeDecode(labels_file,istoken = False)

def getExample(content,label=None, tokenF = lambda x:x.split()):
    '''
    :param content:多个句子组成的文档，为一个字符串
    :param label: 文档的类别信息
    :param tokenF: 对单个句子分词的函数
    :return:
        sen_seq_token是一个list，其元素也是list，[[doc1_w1,doc1_w2,doc1_w3,...], [doc2_w1,doc2_w2,doc2_w3,...], ...]
        sen_seq_len是该文档包含的句子条数
    '''
    sentences = list(SentenceSplitter.split(content))
    sen_seq_len = []
    sen_seq_token = []
    for ss in sentences:
        ss = tokenF(ss)
        ss_len = len(ss)
        if ss_len == 0: continue
        sen_seq_token.append(ss)  #[tokens_list_sen1, tokens_list_sen2, ...]
        sen_seq_len.append(len(ss))  # [len(sen1), len(sen2), ... ]
    return (sen_seq_token,len(sen_seq_len),sen_seq_len,label)

def createDatasets(fn='train_data/labor_dispute_dev.csv'):
    examples = []
    csv_reader = csv.reader(open(fn,'r'),delimiter='\t')
    examples = [getExample(ex[1],ex[0]) for ex in csv_reader]
    return examples

def pad(examples):
    '''
    :param examples:  is a list, each element is the returned value of getExample
    :return:  transform the examples to a 2-d matrix,which is the first returned value,each row is a sentence
            the second returned value: is the label list
            the third returned value: is a list, which records the num of sentences in each document;
            the forth returned value: is a list, which records the length of tokens in each sentences

    '''
    labels = []
    sentences_len_seq, tokens_len_seq = [],[]
    for _, sen_num, seq_len, label in examples:
        if label is not None:
            labels.append(label2idx[label])
        sentences_len_seq.append(sen_num)
        tokens_len_seq.extend(seq_len)

    max_length_token = np.max(tokens_len_seq)

    #[total_sen_num,token_len]
    batch_matrix = np.zeros((np.sum(sentences_len_seq),max_length_token),np.int)

    row_idx = 0
    for ex in examples:
        sentences = ex[0]
        for sen in sentences:
            for j,token in enumerate(sen):
                batch_matrix[row_idx,j] = token2idx.get(token,1)
            row_idx += 1
    return torch.from_numpy(batch_matrix),torch.from_numpy(np.array(labels,np.int)),\
           torch.from_numpy(np.array(sentences_len_seq,np.int)),\
           torch.from_numpy(np.array(tokens_len_seq,np.int))

def genBatch(examples,batch_size=16,istrain=True):
    # pdb.set_trace()
    #执行一遍，跑一个epoch
    if istrain:
        random.shuffle(examples)
    for step in range(0,len(examples),batch_size):
        mini_batch = examples[step:step+batch_size]
        paded_batch,label_patch,sentences_len_seq,tokens_len_seq = pad(mini_batch)
        yield paded_batch,label_patch, sentences_len_seq, tokens_len_seq

# the inference for model

def eval(model,dev_dataset,batch_size,is_cuda):
    model.eval()
    acc = 0.0
    avg_loss = 0.0
    step = 0
    for batch, label, sentence_len_seq, tokens_len_seq in genBatch(dev_dataset, batch_size=batch_size,istrain=False):
        if is_cuda:
            batch, label, sentence_len_seq, tokens_len_seq = batch.cuda(), label.cuda(), sentence_len_seq.cuda(), tokens_len_seq.cuda()

        logit = model.forward(batch, sentence_len_seq, tokens_len_seq)
        avg_loss += F.cross_entropy(logit,label)
        pre_tmp = torch.max(logit, 1)[1].view(label.size()).data
        acc += (pre_tmp == label.data).sum()
        step += 1
        # print(step,pre_tmp)

    # print(avg_loss,acc,step)
    acc = float(acc) /len(dev_dataset)
    avg_loss = float(avg_loss)/ step
    print('Evalution, avg_loss:{:.6f}, acc:{:.4f}'.format(avg_loss, acc))
    return acc, avg_loss

def load_model(model,checkpoints_dir):
    try:
        with open(os.path.join(checkpoints_dir,'checkpoint')) as fp:
            model_name = fp.readline().strip()
            model_name = os.path.join(checkpoints_dir,model_name)
            # pdb.set_trace()
            para_dict = torch.load(model_name)

            model.load_state_dict(para_dict,True)
            print('Load model weight from {}'.format(model_name))
    except:
        print('Load model weight failed')

def save_model(model,checkpoints_dir,save_prefix):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    model_name = '{}.pkl'.format(save_prefix)
    save_path = os.path.join(checkpoints_dir,model_name)
    torch.save(model.state_dict(),save_path)
    with open(os.path.join(checkpoints_dir,'checkpoint'),'w') as fp:
        fp.write(model_name)
    print('Save model weight to {}'.format(save_path))

def train(model,cfg):
    '''
    :param model:
    :param train_dataset:
            the data for training the model, which is a list,
            one element likes: batch*num_sentence, label_batch,sentence_len_seq, tokens_len_seq
    :param checkpoints_dir: for loading or saving weights
    :return:
    '''
    train_dataset = createDatasets(cfg.train_data_file)
    dev_dataset = createDatasets(cfg.dev_data_file)

    is_cuda = next(model.parameters()).is_cuda
    load_model(model,cfg.checkpoints_dir)
    model.train()
    optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=cfg.lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                          lr=cfg.lr,momentum = cfg.momentum)
    # pdb.set_trace()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    # for i in range(20):
    #     scheduler.step()
    # print(scheduler.get_lr())
    # pdb.set_trace()

    step = 0
    last_loss = 0
    last_step = step
    dev_acc = 0.0

    for epoch in range(cfg.epochs):
        scheduler.step()

        for batch,label,sentence_len_seq,tokens_len_seq in genBatch(train_dataset,batch_size=cfg.batch_size):

            if is_cuda:
                batch,label,sentence_len_seq,tokens_len_seq = batch.cuda(),label.cuda(),sentence_len_seq.cuda(),tokens_len_seq.cuda()

            optimizer.zero_grad()
            logit = model.forward(batch,sentence_len_seq,tokens_len_seq)
            loss = F.cross_entropy(logit,label)
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.display_step == 0:
                # pdb.set_trace()
                corrects = (torch.max(logit, 1)[1].view(label.size()).data == label.data)
                corrects = corrects.sum()
                acc = float(corrects) / label.size(0)
                print('epoch:{},step:{},loss:{:.6f},acc:{:.4f},lr:{}'.format(epoch, step, float(loss.data), acc,scheduler.get_lr()))

        if epoch % cfg.save_epoch == 0 and epoch>0:
            acc, avg_loss = eval(model,dev_dataset,cfg.batch_size,is_cuda)
            model.train()
            if acc > dev_acc:
                prefix = 'HAN_epoch_{}_acc_{:.4f}_avg_loss_{:.6f}'.format(epoch,acc, avg_loss)
                save_model(model, cfg.checkpoints_dir, prefix)
                dev_acc = acc

def infer(model,docment_list,batch_size=1):
    examples = []
    for doc in docment_list:
        ex = getExample(doc,None,tokenF=lambda x: list(jieba.cut(x)))
        examples.append(ex)

    predicts = []
    for batch,_,sentence_len_seq,tokens_len_seq in genBatch(examples,batch_size=batch_size,istrain=False):
        if model.use_gpu:
            batch, sentence_len_seq, tokens_len_seq = batch.cuda(), sentence_len_seq.cuda(), tokens_len_seq.cuda()
        logit = model.forward(batch,sentence_len_seq,tokens_len_seq)
        # pdb.set_trace()
        logit = torch.softmax(logit,1)
        predict = torch.max(logit, 1)[1]
        predicts.extend([(idx2label[int(a)],float(logit[i,a].data)) for i,a in enumerate(predict)])
    return predicts
