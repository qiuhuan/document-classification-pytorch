
import util
from model import HAN
import config as cfg

if __name__ == '__main__':
    vocab_size = len(util.token2idx)
    num_classes = len(util.label2idx)
    m = HAN(num_classes,vocab_size,embed_dim=cfg.embed_dim,
            w_hidden_size = cfg.w_hidden_size,     s_hidden_size = cfg.s_hidden_size,
            w_num_layers = cfg.w_num_layers,       s_num_layers=cfg.s_num_layers,
            w_bidirectional = cfg.w_bidirectional, s_bidirectional = cfg.s_bidirectional,
            w_attention_dim=cfg.w_attention_dim,   s_attention_dim=cfg.s_attention_dim,
            use_gpu = cfg.use_gpu)

    if cfg.mode =='train':
        util.train(m,cfg)
    elif cfg.mode == 'eval':
        dev_dataset = util.createDatasets(cfg.dev_data_file)
        util.load_model(m, cfg.checkpoints_dir)
        m.eval()
        util.eval(m,dev_dataset,cfg.batch_size,True)
    else:
        text = open('test_data/aaa.txt','r').readlines()
        text = ''.join(text)
        text = text.replace('\n','').replace(' ','').replace('\t','')
        util.load_model(m, cfg.checkpoints_dir)
        m.eval()
        res = util.infer(m,[text])
        print('predicts are ',res)

    print('done')