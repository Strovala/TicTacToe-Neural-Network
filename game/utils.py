def map_to_char(number):
    if number == 1:
        return 'x'
    if number == -1:
        return 'o'
    return ' '


def update_dataset(func):
    def wrapper(*args):
        self, player, action = args

        prev_state = self.board.copy()
        prev_hash = '{} {}'.format(
            player.code, ' '.join(map(str, map(int, prev_state)))
        )
        prev_value = self.dataset.get(prev_hash)
        if prev_value is None:
            self.dataset[prev_hash] = {}
        func(*args)
        next_state = self.board.copy()
        next_hash = '{} {}'.format(
            player.code, ' '.join(map(str, map(int, next_state)))
        )
        next_value = self.dataset[prev_hash].get(next_hash)
        if next_value is None:
            self.dataset[prev_hash][next_hash] = 0
    return wrapper
