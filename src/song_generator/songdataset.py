import regex as re
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
