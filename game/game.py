import numpy as np
import pandas as pd
from game.utils import map_to_char


class Game(object):
    def __init__(self, height, width):
        self.board = np.zeros(height * width)
        self.height = height
        self.width = width
        self.winner = 0
        self.dataset = {}

    def reset(self):
        self.board = np.zeros(self.height * self.width)
        self.winner = 0
        self.dataset = {}

    def render(self):
        print('=====')
        rows = np.array_split(self.board, self.height)
        for row in rows[:-1]:
            row = map(map_to_char, row)
            print('|'.join(row))
            print('-----')
        row = rows[-1]
        row = map(map_to_char, row)
        print('|'.join(row))
        print('=====')

    @property
    def legals(self):
        raise NotImplemented()

    @property
    def end(self):
        raise NotImplemented()

    def update(self, player, action):
        raise NotImplemented()


class TicTacToe(Game):
    def __init__(self, height, width):
        super(TicTacToe, self).__init__(height, width)

    @property
    def legals(self):
        # np.where returns tuple becouse it
        # supose to be used for indexing
        return np.where(self.board == 0)[0]

    @property
    def end(self):
        def _check_rows(rows):
            for row in rows:
                if np.all(row[0] != 0 and row == row[0]):
                    return row[0]
            return None
        # Check draw
        if not np.any(self.legals):
            # self.winner is already 0
            return True
        # Check rows
        rows = self.board.reshape(self.height, self.width)
        winner = _check_rows(rows)
        if winner:
            self.winner = winner
            return True
        # Check columns
        cols = np.rot90(rows)
        winner = _check_rows(cols)
        if winner:
            self.winner = winner
            return True
        # Check diagonals
        diagonals = [np.diag(rows), np.diag(np.fliplr(rows))]
        winner = _check_rows(diagonals)
        if winner:
            self.winner = winner
            return True
        return False

    def update(self, player, action):
        prev_state = self.board.copy()
        prev_hash = '{} {}'.format(
            player.code, ' '.join(map(str, map(int, prev_state)))
        )
        prev_value = self.dataset.get(prev_hash)
        if prev_value is None:
            self.dataset[prev_hash] = {}
        self.board[action] = player.code
        next_state = self.board.copy()
        next_hash = '{} {}'.format(
            player.code, ' '.join(map(str, map(int, next_state)))
        )
        next_value = self.dataset[prev_hash].get(next_hash)
        if next_value is None:
            self.dataset[prev_hash][next_hash] = 0


class Player(object):
    def __init__(self, code, game):
        self.code = code
        self.game = game

    def move(self):
        pass


class Human(Player):
    def __init__(self, code, game):
        super(Human, self).__init__(code, game)

    def move(self):
        return int(input())


class Randomko(Player):
    def __init__(self, code, game):
        super(Randomko, self).__init__(code, game)

    def move(self):
        return np.random.choice(self.game.legals)


class Datasetko(Player):
    def __init__(self, code, game):
        super(Datasetko, self).__init__(code, game)
        self.df = pd.DataFrame.from_csv('ttt_dataset.csv')

    def move(self):
        board_hash = '{} {}'.format(
            self.code, ' '.join(map(str, map(int, self.game.board)))
        )
        df_one = self.df.loc[self.df['current'] == board_hash]
        if df_one.empty:
            return np.random.choice(self.game.legals)
        prev_board, next_board, winning = df_one.iloc[0]
        prev_board = prev_board.split()[1:]
        next_board = next_board.split()[1:]
        for i in range(len(prev_board)):
            if prev_board[i] != next_board[i]:
                print("Confidence:", winning)
                return i
        print('WARNING: prev and next board same')
        return np.random.choice(self.game.legals)
