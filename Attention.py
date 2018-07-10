import argparse
import cv2
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import training
from chainer import serializers
from chainer import Variable

class SoftAttention(chainer.Chain):
    def __init__(self, hidden_size, bidirection=True, gpu=False):
        self.hidden_size = hidden_size
        self.bidirection = bidirection
        self.gpu = gpu

        if gpu:
            import cupy as cp
            self.xp = cp
        else:
            import numpy as np
            self.xp = np

        with self.init_scope():
            if bidirection:
                self.bh = L.Linear(hidden_size, hidden_size)
            self.fh = L.Linear(hidden_size, hidden_size)
            self.hh = L.Linear(hidden_size, hidden_size)
            self.hw = L.Linear(hidden_size, 1)

    def UniDirectionalAttention(self, fh, h):
        batch_size = h.shape[0]
        ws = []

        for f, h in zip(fh, h):
            hw = self.hw(F.tanh(self.fh(f) + self.hh(h)))
            ws.append(hw)

        context = F.softmax(hw)
        att_f = Variable(self.xp.zeros(batch_size, self.hidden_size), dtype='float32')

        for f, w in zip(f, context):
            att_f += F.reshape(F.matmul(f, w), (batch_size, self.hidden_size))

        return att_f

    def BiDirectionalAttention(self, fh, bh, h):
        batch_size = h.shape[0]
        ws = []

        for f, b, h in zip(fh, bh, h):
            hw = self.hw(F.tanh(self.fh(f) + self.bh(b) + self.hh(h)))
            ws.append(hw)

        context = F.softmax(hw)
        att_f = Variable(self.xp.zeros(batch_size, self.hidden_size), dtype='float32')
        att_b = Variable(self.xp.zeros(batch_size, self.hidden_size), dtype='float32')

        for f, b, w in zip(fh, bh, context):
            att_f += F.reshape(F.matmul(f, w), (batch_size, self.hidden_size))
            att_b += F.reshape(F.matmul(b, w), (batch_size, self.hidden_size))

        return att_f, att_b

    def __call__(self, **args):
        if self.bidirection:
            return self.BiDirectionalAttention(args[0], args[1], args[2])
        else:
            return self.UniDirectionalAttention(args[0], args[1])
