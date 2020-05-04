from fastNLP.io import Pipe
from fastNLP.io import DataBundle
from fastNLP.io.pipe.utils import _add_words_field, _indexize
from fastNLP.io.pipe.utils import iob2, iob2bioes

from fastNLP.io import Conll2003NERLoader
from fastNLP import Const

def word_shape(words):
    shapes = []
    for word in words:
        caps = []
        for char in word:
            caps.append(char.isupper())
        if all(caps):
            shapes.append(0)
        elif any(caps) is False:
            shapes.append(1)
        elif caps[0]:
            shapes.append(2)
        elif any(caps):
            shapes.append(3)
        else:
            shapes.append(4)
    return shapes

class VLSP2016NERPipe(Pipe):
    def __init__(self, encoding_type: str = 'bio', lower: bool = False, word_shape: bool=False):
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle) -> DataBundle:
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')
        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                field_name=Const.INPUT, new_field_name=Const.INPUT)

        # index
        _indexize(data_bundle)
        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = Conll2003NERLoader().load(paths)
        
        data_bundle = self.process(data_bundle)
        return data_bundle