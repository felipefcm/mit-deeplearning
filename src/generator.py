import numpy as np
import torch
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import song_generator.abclib as abclib
import song_generator.songdataset as songdataset

from song_generator.network import SongNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()

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

net = net.to(device)

optimiser = torch.optim.Adam(net.parameters(), lr=0.01)


def take_predicted_sample(output):
    categorical = torch.distributions.Categorical(logits=output)
    sample = categorical.sample()
    chars = ''.join([idx2char[idx] for idx in sample])
    return chars


def print_predicted_sample(chars, input):
    print()
    print('input', ''.join([idx2char[idx] for idx in input]))
    print('predict', chars)
    print()


num_epochs = 450
for epoch in range(num_epochs):
    x, y = songdataset.get_batch(
        vectorised_songs,
        seq_length=400,
        batch_size=42
    )

    x = x.to(device)
    y = y.to(device)

    net.train()
    output = net(x)

    output_p = output.permute(0, 2, 1)
    loss = F.nll_loss(output_p, y)
    print(f'epoch {epoch} loss', loss.item())
    writer.add_scalar('Loss/mean', loss.item(), epoch)

    optimiser.zero_grad()

    loss.backward()
    optimiser.step()

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        net.eval()
        with torch.no_grad():
            chars = take_predicted_sample(output[0])
            writer.add_text('Predicted', f'loss={loss.item()}\n{chars}', epoch)

        time.sleep(8)


writer.close()
