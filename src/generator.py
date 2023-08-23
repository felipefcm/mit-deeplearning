import numpy as np
import torch
from torch.utils.data import DataLoader

import song_generator.abclib as abclib
import song_generator.songdataset as songdataset

from song_generator.network import SongNetwork

# dataset = songdataset.SongDataset('./data/irish.abc')

# dataloader = DataLoader(
#     dataset=dataset
# )

songs = songdataset.load_training_data('./data/irish.abc')
print('loaded')

songs_joined = '\n\n'.join(songs)
vocab = sorted(set(songs_joined))

char2idx = {i: c for c, i in enumerate(vocab)}
idx2char = np.array(vocab)


def vectorise_string(string: str):
    return [
        char2idx.get(char) for char in string
    ]


vectorised_songs = vectorise_string(songs_joined)

net = SongNetwork(
    rnn_units=1024,
    vocab_size=len(vocab),
    embedding_dim=256
)

x, y = songdataset.get_batch(vectorised_songs, seq_length=100, batch_size=32)
print(net(x))
print(net(x))

print('a')
