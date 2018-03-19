import numpy as np
import pandas as pd
from game.utils import map_to_char, update_dataset


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
        separator_length = self.width * 2
        print('=' * separator_length)
        rows = np.array_split(self.board, self.height)
        for row in rows[:-1]:
            row = map(map_to_char, row)
            print('|'.join(row))
            print('-' * separator_length)
        row = rows[-1]
        row = map(map_to_char, row)
        print('|'.join(row))
        print('=' * separator_length)

    @property
    def legals(self):
        raise NotImplemented()

    @property
    def end(self):
        """
        This method needs to set self.winner before returning
        """
        raise NotImplemented()

    def update(self, player, action):
        raise NotImplemented()


class ConnectFour(Game):
    def __init__(self, height, width):
        super(ConnectFour, self).__init__(height, width)

    @property
    def legals(self):
        rows = np.array_split(self.board, self.height)
        cols = np.rot90(rows, k=3)
        available = []
        for index, col in enumerate(cols):
            if len(np.where(col == 0)[0]) > 0:
                available.append(index)
        return available

    def _validate(self, board, i, j, match_cnt=4):
        fours = []
        try:
            rows = [board[i+k][j] for k in range(match_cnt)]
            fours.append(rows)
        except Exception:
            pass
        try:
            cols = [board[i][j+k] for k in range(match_cnt)]
            fours.append(cols)
        except Exception:
            pass
        try:
            diagonal_right = [board[i-k][j+k] for k in range(match_cnt)]
            fours.append(diagonal_right)
        except Exception:
            pass
        try:
            diagonal_left  = [board[i+k][j+k] for k in range(match_cnt)]
            fours.append(diagonal_left)
        except Exception:
            pass
        for four in fours:
            four_set = set(four)
            if len(four_set) == 1 and four[0] != 0:
                self.winner = four[0]
                return True
        return False

    @property
    def end(self):
        # Check draw
        if not np.any(self.legals):
            # self.winner is already 0
            return True
        rows = np.array_split(self.board, self.height)
        for i in range(len(rows)):
            for j in range(len(rows[0])):
                is_end = self._validate(rows, i, j)
                if is_end:
                    return True
        return False

    @update_dataset
    def update(self, player, action):
        rows = np.array_split(self.board, self.height)
        cols = np.rot90(rows, k=3)
        cols_index = np.where(cols[action] == 0)[0][0]
        index = (self.height - cols_index - 1) * 7 + action
        self.board[index] = player.code


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
        def check_rows(rows):
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
        winner = check_rows(rows)
        if winner:
            self.winner = winner
            return True
        # Check columns
        cols = np.rot90(rows)
        winner = check_rows(cols)
        if winner:
            self.winner = winner
            return True
        # Check diagonals
        diagonals = [np.diag(rows), np.diag(np.fliplr(rows))]
        winner = check_rows(diagonals)
        if winner:
            self.winner = winner
            return True
        return False

    @update_dataset
    def update(self, player, action):
        self.board[action] = player.code


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
        self.df = None

    def move(self):
        board_hash = '{} {}'.format(
            self.code, ' '.join(map(str, map(int, self.game.board)))
        )
        df_one = self.df.loc[self.df['current'] == board_hash]
        if df_one.empty:
            return None, None, None
        prev_board, next_board, winning = df_one.iloc[0]
        prev_board = prev_board.split()[1:]
        next_board = next_board.split()[1:]
        return prev_board, next_board, winning


class DatasetkoTicTacToe(Datasetko):
    def __init__(self, code, game):
        super(DatasetkoTicTacToe, self).__init__(code, game)
        self.df = pd.DataFrame.from_csv('ttt_dataset.csv')

    def move(self):
        prev_board, next_board, winning = super(
            DatasetkoTicTacToe, self
        ).move()
        if prev_board is None:
            return np.random.choice(self.game.legals)
        for i in range(len(prev_board)):
            if prev_board[i] != next_board[i]:
                print("Confidence:", winning)
                return i
        print('WARNING: prev and next board same')
        return np.random.choice(self.game.legals)


class DatasetkoConnectFour(Datasetko):
    def __init__(self, code, game):
        super(DatasetkoConnectFour, self).__init__(code, game)
        self.df = pd.DataFrame.from_csv('c4_dataset.csv')

    def move(self):
        prev_board, next_board, winning = super(
            DatasetkoConnectFour, self
        ).move()
        if prev_board is None:
            return np.random.choice(self.game.legals)
        for i in range(len(prev_board)):
            if prev_board[i] != next_board[i]:
                print("Confidence:", winning)
                return i % self.game.width
        print('WARNING: prev and next board same')
        return np.random.choice(self.game.legals)
