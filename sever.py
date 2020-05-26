from flask import Flask
from flask import request
from inference import Inference
from fastNLP.core.utils import _build_args
import numpy as np
from flask import jsonify
# from pyvi import ViTokenizer
from vncorenlp import VnCoreNLP
import torch
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)
#thay đường dẫn đến VnCoreNLP-1.1.1.jar trước khi chạy
annotator = VnCoreNLP("./VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
inference = Inference()
model = inference.model
data__ = inference.data
dict_labels = {
    0: 'O',
    1: 'B-PER',
    2: 'B-LOC',
    3: 'I-PER',
    4: 'I-LOC',
    5: 'I-ORG',
    6: 'B-ORG',
    7: 'B-MISC',
    8: 'I-MISC'
}
def predicter(sentence, model):
    chars = []
    # print(sentence)
    for sen in sentence.split():
        idx = data__.to_index(sen)
        if idx == 1:
            idx = data__.to_index(sen.lower())
        # print(idx)
        chars.append(idx)

    seq_len = len(chars)
    target = [0]*seq_len

    target = torch.Tensor(np.array([target]))
    target = target.type(torch.LongTensor)
    seq_len = torch.Tensor(np.array([seq_len]))
    seq_len = seq_len.type(torch.LongTensor)
    chars = torch.Tensor(np.array([chars]))
    chars = chars.type(torch.LongTensor)
    
    z = dict({'target': target, 'seq_len': seq_len, 'chars': chars})
    x = _build_args(model.predict, **z)
    result = model.predict(**x)
    nx = result['pred'][0].numpy()
    result = []
    for i, j in zip(sentence.split(), nx):
        tmp = {}
        tmp['text'] = i
        tmp['value'] = dict_labels[j]
        result.append(tmp)
    return result
@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/tener', methods=['POST'])
def tener():
    text = request.get_data()
    text = text.decode('utf8')
    _text = json.loads(text)
    text = _text['text']
    print(text)
    # print(type(text))
    __text = annotator.tokenize(text)
    _text = []
    for sen in __text:
        _text += sen
    text = " ".join(_text)
    print(text)
    # text = ViTokenizer.tokenize(text)
    _result = predicter(text, model)
    result = {}
    result['sentence'] = _result
    return jsonify(result)

if __name__ == '__main__':
   app.run()

