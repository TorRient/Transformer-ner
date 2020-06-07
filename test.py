from models.TENER import TENER
from fastNLP.embeddings import CNNCharEmbedding
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.io.pipe.conll import OntoNotesNERPipe
from fastNLP.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed
from modules.pipe import VLSP2016NERPipe
from fastNLP.core import Vocabulary
import torch
import argparse
from modules.callbacks import EvaluateCallback
from fastNLP.core.batch import DataSetIter
torch.nn.Module.dump_patches = True
device = None

dataset ='vlsp2016'
n_heads = 14
head_dims = 128
num_layers = 2
lr = 0.0009
attn_type = 'adatrans'
char_type = 'bilstm'

pos_embed = None

#########hyper
batch_size = 16
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True
#########hyper

dropout=0.15
fc_dropout=0.4

encoding_type = 'bio'
name = 'caches/{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, char_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

@cache_results(name, _refresh=False)
def load_data():
    if dataset == 'vlsp2016':
        paths = {'test': "./data_2/test.txt",
                 'train': "./data_2/train.txt",
                 'dev': "./data_2/dev.txt"}
        data = VLSP2016NERPipe(encoding_type=encoding_type).process_from_file(paths)
        vocab = []
        with open("vocab.txt", 'r') as files:
            for word in files:
                vocab.append(word.replace("\n", ""))
        data.get_vocab('words').add_word_lst(vocab)
    char_embed = None
    if char_type == 'cnn':
        char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max'
                                      , include_word_start_end=False, min_char_freq=2)
    elif char_type in ['adatrans', 'naive']:
        char_embed = TransformerCharEmbed(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, pool_method='max', activation='relu',
                 min_char_freq=2, requires_grad=True, include_word_start_end=False,
                 char_attn_type=char_type, char_n_head=3, char_dim_ffn=60, char_scale=char_type=='naive',
                 char_dropout=0.15, char_after_norm=True)
    elif char_type == 'bilstm':
        char_embed = LSTMCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                 dropout=0.3, hidden_size=100, pool_method='max', activation='relu',
                 min_char_freq=2, bidirectional=True, requires_grad=True, include_word_start_end=False)
    word_embed = StaticEmbedding(vocab=data.get_vocab('words'), first_dim=20000,
                                 model_dir_or_name="word2vec",
                                 requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=normalize_embed)
    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed], dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed
    data__ = data.get_vocab('words')
    data.rename_field('words', 'chars')
    return data, embed, data__

class Model():
    def __init__(self):
        self.data_bundle, self.embed, self.data = load_data()
    def get_model(self):
        self.model = TENER(tag_vocab=self.data_bundle.get_vocab('target'), embed=self.embed, num_layers=num_layers,
                        d_model=d_model, n_head=n_heads,
                        feedforward_dim=dim_feedforward, dropout=dropout,
                        after_norm=after_norm, attn_type=attn_type,
                        bi_embed=None,
                        fc_dropout=fc_dropout,
                        pos_embed=pos_embed,
                        scale=attn_type=='transformer')
        #Thay model path trước khi chạy
        model_path = './model/model.bin'
        states = torch.load(model_path).state_dict()
        self.model.load_state_dict(states)
        return self.model, self.data
