import pandas as pd

class Dataset(object):
    dataset = {}
    df = pd.DataFrame(columns=['current', 'next', 'winning'])

    @classmethod
    def update_dataset(cls, game):
        for hash in game.dataset:
            for key in game.dataset[hash]:
                game.dataset[hash][key] = game.winner
        for hash in game.dataset:
            gameset = cls.dataset.get(hash)
            if gameset is None:
                cls.dataset[hash] = game.dataset[hash]
            else:
                for key in game.dataset[hash]:
                    if gameset.get(key) is None:
                        gameset[key] = game.winner
                    else:
                        gameset[key] += game.winner

    @classmethod
    def export(cls):
        curr = 0
        total = len(cls.dataset)
        for hash in cls.dataset:
            player = int(hash.split()[0])
            available = []
            for key in cls.dataset[hash]:
                value = cls.dataset[hash][key]
                available.append((key, value))
            if player > 0:
                key, value = max(available, key=lambda item: item[1])
            else:
                key, value = min(available, key=lambda item: item[1])
            cls.df.loc[curr] = [hash, key, value]
            print('\rCurently processing {0:.2f}%'.format(
                (curr+1)*100/total
            ), end='')
            curr += 1
        cls.df.to_csv('c4_dataset.csv')
