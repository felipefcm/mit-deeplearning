import song_generator.abclib as abclib
import song_generator.songdataset as songdataset

dataset = songdataset.SongDataset('./src/song_generator/irish.abc')

print('loaded')
abclib.play_song(dataset.__getitem__(1))
