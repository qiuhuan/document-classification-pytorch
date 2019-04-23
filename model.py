import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
import pdb
def resize2batchSentences(feature_tensor,sen_len_seq):
    '''
    pad 后的batch size 是 【sum(sen_len_seq), dim】,即 feature_tensor.size
    将其转为[batch, max_sen_len, dim], 不足max_sen_len的，补0
    '''
    dim, max_sen_len = feature_tensor.size(1),torch.max(sen_len_seq)
    # feature_tensor = feature_tensor.float()
    vv, row_idx = [],0
    for length in sen_len_seq:
        length = int(length)
        v = feature_tensor[row_idx:row_idx+length,:]
        if length < max_sen_len:
            patch = torch.zeros(max_sen_len-length,dim)
            if feature_tensor.is_cuda:
                v = torch.cat([v,patch.cuda()],0)
            else:
                v = torch.cat([v,patch],0)
        vv.append(v.unsqueeze(0))
        row_idx+= length
    return torch.cat(vv,0)

class EncodeWithAttention(nn.Module):
    '''
    参考论文Hierarchical Attention Networks for Document Classification
    '''
    def __init__(self,input_size,hidden_size,num_layers=1,bidirectional=True,attention_dim=None):
        super(EncodeWithAttention,self).__init__()

        self.bidirectional = bidirectional

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional = self.bidirectional,
                          num_layers = num_layers,
                          batch_first=True)
        # for attention parameters
        num_directions = 2 if self.bidirectional else 1
        if attention_dim is None:
            attention_dim = num_directions*hidden_size

        self.project = nn.Linear(num_directions*hidden_size,attention_dim,bias = True)
        self.context = nn.Parameter(torch.Tensor(attention_dim,1))
        # pdb.set_trace()
        #context initial
        self.context.data.uniform_(-1,1)



    def forward(self,input,seq_len,use_gpu=True):
        '''
        :param input: [B,T,input_size]
        :param seq_len: the seq for time step
        :return:
        ============ K == num_directions*hidden_size
        '''
        _,idx_sort = torch.sort(seq_len,descending=True)
        _,idx_unsort = torch.sort(idx_sort,descending = False)
        #encode

        input_sort = input[idx_sort,:,:]
        seq_len_sort = seq_len[idx_sort]
        rnn_input = pack_padded_sequence(input_sort,seq_len_sort,batch_first=True)
        rnn_output,h = self.rnn(rnn_input)  #[B,T,input_size]  ->  [B,T,K]
        rnn_output = pad_packed_sequence(rnn_output,batch_first=True)
        rnn_output = rnn_output[0][idx_unsort,:,:]

        #attention
        u = torch.tanh(self.project(rnn_output)) #论文中公式5 [B,T,K]->[B,T,attention_dim]
        #[B,T,attention_dim]*[attention_dim,1]->[B,T,1] ->[B,T]
        att_weight = torch.cat([(u[i,:,:].mm(self.context)).unsqueeze(0) for i in range(u.size(0))],0)
        att_weight = att_weight.squeeze(2)
        att_weight = F.softmax(att_weight,dim=1).unsqueeze(1)  #[B,T] -> [B,1,T]
        output = torch.bmm(att_weight,rnn_output).squeeze(1)  #[B,1,T]*[B,T,K] -> [B,1,K] -> [B,K]
        return output


class HAN(nn.Module):
    '''
    参考论文Hierarchical Attention Networks for Document Classification
    '''
    def __init__(self,num_classes,vocab_size,embed_dim=200,
                 w_hidden_size=50,s_hidden_size=50,
                 w_num_layers = 1,s_num_layers = 1,
                 w_bidirectional=True,s_bidirectional=True,
                 w_attention_dim=None,
                 s_attention_dim=None, use_gpu=True):
        super(HAN,self).__init__()

        self.use_gpu = use_gpu

        self.embed = nn.Embedding(vocab_size,embed_dim)

        self.word_encode = EncodeWithAttention(embed_dim,w_hidden_size,
                                               num_layers=w_num_layers,
                                               bidirectional=w_bidirectional,
                                               attention_dim=w_attention_dim)

        self.sentences_encode = EncodeWithAttention(2*w_hidden_size,s_hidden_size,
                                                    num_layers=s_num_layers,
                                                    bidirectional=s_bidirectional,
                                                    attention_dim=s_attention_dim)
        self.fc1 = nn.Linear(2*s_hidden_size,num_classes)

        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif type(m) == nn.Embedding:
                m.weight.data.uniform_(-0.1,0.1)

        self.apply(init_weight)

        if self.use_gpu:
            self.cuda()

        print(self)



    def forward(self,input,sentences_len_seq, tokens_len_seq):
        '''
        :param input: [sum(sentences_len_seq), max(tokens_len_seq)],用[B*Ts_sum,Tw]表示
        :param sentences_len_seq: 1维Tensor
        :param tokens_len_seq: 1维Tensor
        :return:
        '''
        # print(input)
        embeded = self.embed(input) #[B*Ts,Tw] -> [B*Ts,Tw,vec_dim]
        # print(embeded)
        # [B*Ts_sum,Tw,vec_dim] -> [B*Ts_sum,1,w_hidden_size*2]->[B*Ts_sum,w_hidden_size*2]
        word_en = self.word_encode.forward(embeded,tokens_len_seq)

        # [B*Ts_sum,w_hidden_size*2] -> [B,Ts,w_hidden_size*2], padding 0 if the length is shorter than Ts
        word_en = resize2batchSentences(word_en,sentences_len_seq)

        #[B,Ts,w_hidden_size*2] -> [B,1,s_hidden_size*2] -> [B,s_hidden_size*2]
        sentences_en = self.sentences_encode.forward(word_en,sentences_len_seq)

        logit = self.fc1(sentences_en)
        return logit

if __name__ == '__main__':

    m = HAN(4,40000)
    # print(m)
    # m.cuda()
    # for p in m.parameters():
    #     print(p.is_cuda,p.device)
    # for p in m.named_parameters():
    #     print(p[0],p[1].device,p[1].is_cuda)