from flask import Flask
from flask import request
from fastNLP.core.utils import _build_args
import numpy as np
from flask import jsonify
from vncorenlp import VnCoreNLP
import torch
from flask_cors import CORS
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='NER')
parser.add_argument('--inference', default=True, type=bool,
                    help='inference or test')

args = parser.parse_args()

print("[INFO] loading model")
if args.inference == True:
    from inference import Model
    model = Model().get_model()
else:
    from test import Model
    model, data__ = Model().get_model()

app = Flask(__name__)
CORS(app)

# Thay đường dẫn đến VnCoreNLP-1.1.1.jar trước khi chạy
annotator = VnCoreNLP("./VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

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

def test(sentence, model):
    chars = []
    # print(sentence)
    for sen in tqdm(sentence.split()):
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
    print(result)
    return result

def inference(sentence, model):
    pred, _ = model.predict([sentence])
    result = []
    for idx, text in enumerate(sentence.split()):
        tmp = {}
        tmp['text'] = text
        tmp['value'] = pred[0][idx][text]
        result.append(tmp)
    print(result)
    return result

@app.route('/tener', methods=['POST'])
def tener():
    text = request.get_data()
    text = text.decode('utf8')
    _text = json.loads(text)
    text = _text['text']
    # print(type(text))
    __text = annotator.tokenize(text)
    _text = []
    for sen in __text:
        _text += sen
    text = " ".join(_text)
    print(text)
    if args.inference == True:
        _result = inference(text, model)
    else:
        _result = test(text, model)
    result = {}
    result['sentence'] = _result
    return jsonify(result)

if __name__ == '__main__':
   app.run()
   
annotator.close()