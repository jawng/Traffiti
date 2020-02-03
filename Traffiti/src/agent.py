import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from collections import deque
import random


def get_ann():
    nn = Sequential()
    nn.add(Dense(200, activation='relu', input_shape=(201,)))
    nn.add(Dropout(0.2))
    nn.add(Dense(100, activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Dense(2, activation='linear'))
    nn.compile(RMSprop(lr=0.01), loss='mean_squared_error')
    nn.summary()
    return nn


class Agent:
    def __init__(self):
        self.epoch = 0
        self.model = self._build_model()
        self.gamma = 0.9
        self.exploration = 1
        self.memory = deque(maxlen=20)
        self.batch_size = 2
        self.min_exploration = 0.01
        self.exploration_decay = 0.995

    def _build_model(self):
        model = get_ann()
        return model

    def memorize(self, instance):
        s, a, next_s, r = instance
        self.memory.append((s, a, next_s, r))

    def schedule(self, i, lr):
        if self.epoch == 70:
            lr = lr / 10
        return lr

    def train(self):
        if len(self.memory) >= self.batch_size:
            miniBatch = random.sample(list(self.memory), self.batch_size)
        else:
            miniBatch = list(self.memory)
        for state, action, next_state, reward in miniBatch:
            target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[LearningRateScheduler(self.schedule)])
        if self.exploration > self.min_exploration:
            self.exploration *= self.exploration_decay

    def predict(self, x):
        if np.random.rand() <= self.exploration:
            a = int(np.round(np.random.rand()))
        else:
            res = self.model.predict(x)[0]
            a = np.argmax(res)
        return a

    def load(self, file):
        self.model.load_weights(file)

    def save(self):
        self.model.save_weights('%s.h5' % self.epoch)
