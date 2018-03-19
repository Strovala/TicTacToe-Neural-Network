import logging
from game.dataset import Dataset
from game.utils import map_to_char
from game.game import (
    TicTacToe, Human, Randomko, Datasetko, ConnectFour, DatasetkoConnectFour,
    MonteCarloBot
)

logger = logging.getLogger()
logging.basicConfig(format='%(message)s', level=logging.DEBUG, handlers=[
    logging.FileHandler('logs.log'),
    logging.StreamHandler()
])


def play():
    """
    Global function for playing one match
    :param player_index: Starting player index
    """
    while not game.end():
        player_index = game.current_player(game.board)
        for player in game.players:
            if isinstance(player, MonteCarloBot):
                player.update(game.board)

        player = game.players[player_index]
        valid, action = False, -1
        while not valid:
            action = player.move()
            if action in game.legals():
                valid = True
        game.update(player, action)

        if render:
            game.render()
    if render:
        logger.info('Winner is the player: {}'.format(
            map_to_char(game.winner)
        ))


if __name__ == "__main__":
    # game = TicTacToe(3, 3)
    game = ConnectFour(6, 7)
    matches = 100
    render = False
    # player_x = Randomko(1, game)
    # player_o = Randomko(-1, game)
    render = True
    player_x = Human(1, game)
    player_o = MonteCarloBot(-1, game, time=15, max_moves=1000)
    # So I can tell players[1] and players[-1]
    game.players = [0, player_x, player_o]
    logger.info('Starting!')
    for match_number in range(matches):
        play()
        # Dataset.update_dataset(game)
        game.reset()
        game.switch_first_player()
        logger.info('{}%'.format(match_number*100/matches))
    logger.info('\nFinished!')
    Dataset.export()

