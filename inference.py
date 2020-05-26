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
device = None
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='vlsp2016')

args = parser.parse_args()

dataset = args.dataset

if dataset == 'vlsp2016':
    n_heads = 14
    head_dims = 128
    num_layers = 6
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
    word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                 model_dir_or_name='word2vec/',
                                 requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=normalize_embed)
    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed], dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed
    data__ = data.get_vocab('words')
    data.rename_field('words', 'chars')
    return data, embed, data__, word_embed

data_bundle, embed, data__, word_embed = load_data()

# data__.add_word_lst(["alex_ferguson", "Alex_ferguson", "ALEX_FERGUSON"])
# words = torch.LongTensor([[data__.to_index(word) for word in ["alex_ferguson", "Alex_ferguson", "ALEX_FERGUSON"]]])
# for word in ["alex_ferguson", "Alex_ferguson", "ALEX_FERGUSON"]:
#     print(data__.to_index(word))
# print(words)
# print(word_embed(words))

# print(data_bundle.get_vocab('words'))
model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
                       d_model=d_model, n_head=n_heads,
                       feedforward_dim=dim_feedforward, dropout=dropout,
                        after_norm=after_norm, attn_type=attn_type,
                       bi_embed=None,
                        fc_dropout=fc_dropout,
                       pos_embed=pos_embed,
              scale=attn_type=='transformer')

# model_path = './best_TENER_lstm_glove'

# # model.load_state_dict('models/best_TENER_lstm_glove')
# states = torch.load(model_path).state_dict()
# model.load_state_dict(states)
# sampler = BucketSampler()
# sampler.set_batch_size(batch_size)


class Inference:
    def __init__(self):
        self._get_model()
        self.data = data__
    def _get_model(self):
        self.model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
                        d_model=d_model, n_head=n_heads,
                        feedforward_dim=dim_feedforward, dropout=dropout,
                        after_norm=after_norm, attn_type=attn_type,
                        bi_embed=None,
                        fc_dropout=fc_dropout,
                        pos_embed=pos_embed,
                        scale=attn_type=='transformer')
        model_path = './best_TENER_lstm_glove'
        states = torch.load(model_path).state_dict()
        self.model.load_state_dict(states)
        sampler = BucketSampler()
        sampler.set_batch_size(batch_size)

inference = Inference()
model = inference.model

from fastNLP.core.utils import _build_args
# sentence = "Đức học Đại_Học Bách_Khoa"
# chars = []
# for sen in sentence.split():
#     idx = data__.to_index(sen)
#     if idx == 1:
#         idx = data__.to_index(sen.lower())
#     print(idx)
#     chars.append(idx)

# seq_len = len(chars)
# target = [0]*seq_len

# import numpy as np

# target = torch.Tensor(np.array([target]))
# target = target.type(torch.LongTensor)
# seq_len = torch.Tensor(np.array([seq_len]))
# seq_len = seq_len.type(torch.LongTensor)
# chars = torch.Tensor(np.array([chars]))
# chars = chars.type(torch.LongTensor)

# z = dict({'target': target, 'seq_len': seq_len, 'chars': chars})
# x = _build_args(model.predict, **z)
# result = model.predict(**x)
# print(result)
# print(result['pred'][0])
# for i in result['pred'][0]:
#     print(type(i))

# nx = result['pred'][0].numpy()
# for n in nx:
#     print(n)

