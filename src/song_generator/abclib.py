import os


def save_song_to_abc(song, filename='tmp'):
    save_name = f'{filename}.abc'

    with open(save_name, 'w') as f:
        f.write(song)

    return filename


def abc2wav(abc_file):
    cmd = f'./src/song_generator/bin/abc2wav.sh {abc_file}'
    return os.system(cmd)


# def play_wav(wav_file):
#     return Audio(wav_file)


def play_song(song):
    basename = save_song_to_abc(song)
    ret = abc2wav(basename + '.abc')
