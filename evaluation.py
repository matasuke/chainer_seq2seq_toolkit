import numpy as np
from nltk.translation import bleu_score

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class CalculateBleu(chainer.training.Extension):
    trigger = (1, 'epoch')
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                        for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
                references, hypotheses,
                smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})


