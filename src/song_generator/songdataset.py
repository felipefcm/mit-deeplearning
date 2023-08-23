import numpy as np
import regex as re
import torch
from torch.utils.data import Dataset


class SongDataset(Dataset):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.songs = load_training_data(dataset_path)

    def __getitem__(self, index):
        return self.songs[index]

    def __len__(self):
        return len(self.songs)


def load_training_data(filepath: str):
    with open(filepath, 'r') as f:
        text = f.read()

    songs = extract_song_snippet(text)
    return songs


def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'

    search_results = re.findall(
        pattern, text, overlapped=True, flags=re.DOTALL
    )

    songs = [song[1] for song in search_results]
    return songs


def get_batch(vectorised_songs, seq_length, batch_size):
    n = len(vectorised_songs) - 1
    seq_indices = np.random.choice(n - seq_length, batch_size)

    input_batch = torch.tensor([
        vectorised_songs[idx:idx + seq_length] for idx in seq_indices
    ])

    output_batch = torch.tensor([
        vectorised_songs[idx + 1:idx + 1 + seq_length] for idx in seq_indices
    ])

    return input_batch, output_batch
