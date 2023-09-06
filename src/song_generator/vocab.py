import numpy as np
import torch
import song_generator.songdataset as songdataset

songs = songdataset.load_training_data('./data/irish.abc')

songs_joined = '\n\n'.join(songs)
vocab = sorted(set(songs_joined))
vocab_size = len(vocab)

char2idx = {i: c for c, i in enumerate(vocab)}
idx2char = np.array(vocab)


def vectorise_string(string: str):
    return [
        char2idx.get(char) for char in string
    ]


vectorised_songs = vectorise_string(songs_joined)


def take_predicted_sample(output):
    categorical = torch.distributions.Categorical(logits=output)
    sample = categorical.sample()
    chars = ''.join([idx2char[idx] for idx in sample])
    return chars
