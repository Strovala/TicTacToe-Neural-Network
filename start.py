from game.dataset import Dataset
from game.game import TicTacToe, Human, Randomko, Datasetko
from game.utils import map_to_char


def play(player_index):
    """
    Global function for playing one match
    :param player_index: Starting player index
    """
    while not game.end:
        player = players[player_index]
        valid, action = False, -1
        while not valid:
            action = player.move()
            if action in game.legals:
                valid = True
        game.update(player, action)
        if render:
            game.render()
        player_index *= -1
    if render:
        print('Winner is the player: {}'.format(
            map_to_char(game.winner)
        ))


if __name__ == "__main__":
    game = TicTacToe(3, 3)
    matches = 5
    render = True
    player_x = Human(1, game)
    player_o = Datasetko(-1, game)
    # So I can tell players[1] and players[-1]
    players = [0, player_x, player_o]
    first_player = 1
    print('Starting!')
    for match_number in range(matches):
        play(first_player)
        Dataset.update_dataset(game)
        game.reset()
        first_player *= -1
        print('\r{}%'.format(match_number*100/matches), end='')
    print('\nFinished!')
    Dataset.export()

