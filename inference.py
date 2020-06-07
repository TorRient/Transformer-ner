from simpletransformers.ner import NERModel
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.io.pipe.conll import OntoNotesNERPipe
from fastNLP.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed
from modules.pipe import VLSP2016NERPipe


class Model():
    def __init__(self):
        self.model = NERModel(model_name='model', use_cuda=False, args={'max_seq_length': 256})

    def get_model(self):
        return self.model
