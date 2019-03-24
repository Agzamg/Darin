import os
import sys
import itertools
import tensorflow as tf
from time import time
import keras
from keras.models import load_model
import abc
import numpy as np
import subprocess

idx2chr = 'abcdefghjklmnop'
chr2idx = {char: idx for idx, char in enumerate(idx2chr)}


def to_move(pos):
    return idx2chr[pos[1]] + str(pos[0] + 1)


def to_pos(move):
    return int(move[1:]) - 1, chr2idx[move[0]]


def list_positions(board, player):
    return np.vstack(np.nonzero(board == player)).T


def sequence_length(board, I, J, value):
    length = 0
    for i, j in zip(I, J):
        if board[i, j] != value:
            break
        length += 1

    return length


def check_horizontal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j + 1, min(j + 5, 15)),
        player
    )

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j - 1, max(j - 5, -1), -1),
        player
    )

    return length >= 5


def check_vertical(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + 5, 15)),
        itertools.repeat(j),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - 5, -1), -1),
        itertools.repeat(j),
        player
    )

    return length >= 5


def check_main_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + 5, 15)),
        range(j + 1, min(j + 5, 15)),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - 5, -1), -1),
        range(j - 1, max(j - 5, -1), -1),
        player
    )

    return length >= 5


def check_side_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i - 1, max(i - 5, -1), -1),
        range(j + 1, min(j + 5, 15)),
        player
    )

    length += sequence_length(
        board,
        range(i + 1, min(i + 5, 15)),
        range(j - 1, max(j - 5, -1), -1),
        player
    )

    return length >= 5


def check(board, pos):
    if not board[pos]:
        return False

    return check_vertical(board, pos) \
        or check_horizontal(board, pos) \
        or check_main_diagonal(board, pos) \
        or check_side_diagonal(board, pos)


class Node():
    def __init__(self, board, color, black_model, white_model):
        self._color = color
        self._black_model = black_model
        self._white_model = white_model

        if color == 'black':
            with black_model[1].as_default():
                self._P = black_model[0].predict(board.reshape(
                    1, 15, 15, 1), batch_size=1, verbose=0)[0]
        else:
            with white_model[1].as_default():
                self._P = white_model[0].predict(board.reshape(
                    1, 15, 15, 1), batch_size=1, verbose=0)[0]
        self._visited = 0
        self._N = np.zeros(225, dtype=np.float32)
        self._R = np.zeros(225, dtype=np.float32)
        self._children = [None for i in range(225)]

    def travail(self, board, move):
        parsed_move = np.unravel_index(move, (15, 15))
        board[parsed_move] = 1
        next_color = 'white'
        if (self._color == 'white'):
            board[parsed_move] = -1
            next_color = 'black'

        child = Node(board, next_color, self._black_model, self._white_model)
        self._children[move] = child


class MCTS():
    def __init__(
            self,
            name,
            black_model,
            white_model,
            black_playout,
            white_playout,
            color=None,
            high=10,
            gamma=1.0,
            samples=None,
            timeout=None,
            verbose=0,
            min_prob=0.8,
            param1=0.25,
            param2=0.85):
        self._name = name
        self._node_model_black = black_model
        self._node_model_white = white_model
        self._black_model = black_playout[0]
        self._black_graph = black_playout[1]
        self._white_model = white_playout[0]
        self._white_graph = white_playout[1]
        self._timeout = timeout
        self._samples = 100000
        self._high = high
        self._color = color
        self._verbose = verbose
        self._gamma = gamma
        self._iters = 0.0
        self._root = None
        self._board = None
        self._start_time = None
        self._min_prob = min_prob
        self._param1 = param1
        self._param2 = param2

    def name(self):
        return self._name

    def playout(self, board, temp_high, color):
        gamma = self._gamma

        if (len(list_positions(board, 0)) == 0):
            return 0

        temp_predictions = 0
        if color == 'black':
            with self._black_graph.as_default():
                temp_predictions = self._black_model.predict(
                    board.reshape(1, 15, 15, 1), batch_size=1, verbose=0)[0]
        else:
            with self._white_graph.as_default():
                temp_predictions = self._white_model.predict(
                    board.reshape(1, 15, 15, 1), batch_size=1, verbose=0)[0]

        temp_pos = np.random.choice(225, p=temp_predictions)
        temp_parsed_pos = np.unravel_index(temp_pos, (15, 15))

        if (board[temp_parsed_pos] != 0):
            if (temp_high % 2):
                return gamma ** (temp_high - 1)
            else:
                return -gamma ** (temp_high - 1)

        board[temp_parsed_pos] = 1
        next_color = 'white'
        if color == 'white':
            next_color = 'black'
            board[temp_parsed_pos] = -1

        if (check(board, temp_parsed_pos)):
            if (temp_high % 2):
                return -gamma ** (temp_high - 1)
            else:
                return gamma ** (temp_high - 1)

        if (temp_high < self._high):
            return self.playout(board, temp_high + 1, next_color)
        return 0

    def simulation(self, ucb_eps=0.01):
        temp_high = -1
        reward = 0
        path = []
        temp_root = self._root
        gamma = self._gamma
        board = np.copy(self._board)

        while temp_high < self._high:
            temp_high += 1
            if (temp_root._visited):
                ucb = ucb_eps * \
                    np.sqrt(2 * np.log(temp_root._N + 1) / (1 + self._iters + temp_root._N.sum()))
                values = (temp_root._R + 10 * temp_root._P) / (1 + temp_root._N) + (temp_root._R >
                                                                                    self._param1 * temp_root._N) + (temp_root._R > self._param2 * temp_root._N)
                temp_pos = np.argmax(values + ucb)
                temp_parsed_pos = np.unravel_index(temp_pos, (15, 15))

                path.append(temp_pos)

                if (board[temp_parsed_pos] != 0):
                    if (temp_high % 2):
                        return path, gamma ** (temp_high - 1)
                    else:
                        return path, -gamma ** (temp_high - 1)

                if (temp_root._color == 'black'):
                    board[temp_parsed_pos] = 1
                else:
                    board[temp_parsed_pos] = -1

                if (check(board, temp_parsed_pos)):
                    if (temp_high % 2):
                        return path, -gamma ** (temp_high - 1)
                    else:
                        return path, gamma ** (temp_high - 1)

                if not temp_root._children[temp_pos]:
                    temp_root.travail(board, temp_pos)

                temp_root = temp_root._children[temp_pos]
            else:
                temp_root._visited = 1

                temp_pos = np.random.choice(225, p=temp_root._P)
                temp_parsed_pos = np.unravel_index(temp_pos, (15, 15))
                path.append(temp_pos)

                if (board[temp_parsed_pos] != 0):
                    if (temp_high % 2):
                        return path, gamma ** (temp_high - 1)
                    else:
                        return path, -gamma ** (temp_high - 1)

                board[temp_parsed_pos] = 1
                next_color = 'white'
                if temp_root._color == 'white':
                    next_color = 'black'
                    board[temp_parsed_pos] = -1

                if (check(board, temp_parsed_pos)):
                    if (temp_high % 2):
                        return path, -gamma ** (temp_high - 1)
                    else:
                        return path, gamma ** (temp_high - 1)

                if not temp_root._children[temp_pos]:
                    temp_root.travail(board, temp_pos)

                if (temp_high < self._high):
                    return path, self.playout(board, temp_high + 1, next_color)
                return path, 0

        return path, reward

    def backprop(self, path, reward):
        current = self._root
        for action in path:
            if current._color == self._color:
                current._R[action] += reward
            else:
                current._R[action] -= reward
            current._N[action] += 1
            current = current._children[action]

        del path

    def dfs(self):
        self._iters = 0
        while (time() - self._start_time <
               self._timeout and self._iters < self._samples):
            self._iters += 1
            path, reward = self.simulation()
            self.backprop(path, reward)
        return self._iters

    def move_legit(self, move):

        if not self._root._children[move]:
            return False

        self._root = self._root._children[move]
        return True

    def predict(self, board, positions):
        self._start_time = time()
        if not self._color:
            if ((225 - len(list_positions(board, 0))) % 2 == 1):
                self._color = 'white'
            else:
                self._color = 'black'

        if (len(list_positions(board, 0)) == 225):
            return np.unravel_index(112, (15, 15))

        done = True
        if (self._root and len(list_positions(board, 0)) < 224):
            for move in positions[-2:]:
                pos = to_pos(move)
                if not self.move_legit(pos[0] * 15 + pos[1]):
                    done = False
        else:
            done = False

        checker = np.zeros(225)
        available = np.zeros(225)
        self._board = np.copy(board)
        for parsed_pos in list_positions(self._board, 0):
            pos = parsed_pos[0] * 15 + parsed_pos[1]
            parsed_pos = tuple(parsed_pos)

            self._board[parsed_pos] = 1
            if (check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = -1
            if (check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = 0
            available[pos] = 1

        if not done:
            self._root = Node(
                self._board,
                self._color,
                self._node_model_black,
                self._node_model_white)

        self.dfs()
        values = (self._root._N > self._iters / 5) * self._root._R / \
            (1 + self._root._N) * available * (1 + 10 * checker)

        if np.max(values) > self._min_prob:
            code_move = np.argmax(values)
        else:
            code_move = np.argmax(
                (self._root._N) * available * (1 + 10 * checker))

        return np.unravel_index(code_move, (15, 15))


if __name__ == "__main__":
    black = load_model('Agz224.h5')
    black_graph = tf.get_default_graph()
    white = load_model('Agz224.h5')
    white_graph = tf.get_default_graph()
    mcts = MCTS(
        name='AZINO777',
        black_model=(
        black,
        black_graph),
        white_model=(
        white,
        white_graph),
        black_playout=(
        black,
        black_graph),
        white_playout=(
        white,
        white_graph),
        timeout=2.75,
        high=15,
        gamma=0.99,
        verbose=1,
        min_prob=0.8,
        param1=0.2,
        param2=0.65)
    board = np.zeros((15, 15))
    while True:
        game = sys.stdin.readline()
        if game:
            point = 1
            if not game == "\n":
                for move in game.split(' '):
                    board[to_pos(move)] = point
                    point *= -1
            prediction = mcts.predict(board, game.split(' '))
            sys.stdout.write(to_move(prediction) + '\n')
            sys.stdout.flush()
        else:
            break
