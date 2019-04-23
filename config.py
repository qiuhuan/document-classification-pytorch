use_gpu         = True
mode            =  'eval'#'train'             #train or eval or infer
checkpoints_dir = 'checkpoints'       #the dir to save mode weights
vocabulary_file = 'train_data/vocabulary.txt'
labels_file     = 'train_data/labels.txt'
log_file        = checkpoints_dir+'/log.txt'

#the parameters for data
train_data_file = 'train_data/labor_dispute_train.csv'
dev_data_file   = 'train_data/labor_dispute_dev.csv'

# parameters for train
epochs          = 40
batch_size      = 64
save_epoch      = 1
display_step    = 100

lr              = 0.001
momentum        = 0.9
# for adjusting lr
step_size       = 5
gamma           = 0.1

#hyper parameters of model
embed_dim       = 200
w_hidden_size   = 50   #the hidden_size of the word encoder
w_num_layers    = 1    #the num_layers of the word encoder
w_bidirectional = True #the bidirectional of the GRU applied in word encoder
w_attention_dim = 2*50 #'the hidden_size of the word encoder')

s_hidden_size   = 50   #the hidden_size of the sentences encoder
s_num_layers    = 1    #the num_layers of the sentences encoder
s_bidirectional = True #the bidirectional of the GRU applied in sentences encoder
s_attention_dim = 2*50 #'the hidden_size of the sentences encoder')




