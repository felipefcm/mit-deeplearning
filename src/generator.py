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


rnn_units = 2048
embedding_dim = 256
lr = 0.005
num_epochs = 1000
seq_length = 500
batch_size = 64

writer = SummaryWriter(
    log_dir=f'runs/lr{lr}_bs{batch_size}_sl{seq_length}_rnn{rnn_units}'
)

net = SongNetwork(
    rnn_units=rnn_units,
    vocab_size=len(vocab),
    embedding_dim=embedding_dim
)

net = net.to(device)
optimiser = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimiser, step_size=100, gamma=0.8
)

hparam_dict = {
    'epochs': num_epochs,
    'lr': lr,
    'rnn_units': rnn_units,
    'embedding_dim': embedding_dim,
    'batch_size': batch_size,
    'seq_length': seq_length,
}

losses = []

for epoch in range(num_epochs):
    x, y = songdataset.get_batch(
        vectorised_songs,
        seq_length=seq_length,
        batch_size=batch_size
    )

    x = x.to(device)
    y = y.to(device)

    net.train()
    output = net(x)

    output_p = output.permute(0, 2, 1)
    loss = F.nll_loss(output_p, y)
    print(f'epoch {epoch} loss', loss.item())

    writer.add_scalar('Loss/mean', loss.item(), epoch)
    losses.append(loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    lr_scheduler.step()

    if epoch % 100 == 0 or epoch == num_epochs - 1:
        net.eval()
        with torch.no_grad():
            chars = take_predicted_sample(output[0])
            writer.add_text('Predicted', f'loss={loss.item()}\n{chars}', epoch)

    if (epoch + 1) % 100 == 0:
        time.sleep(2)

writer.add_hparams(
    hparam_dict,
    {'hparam/loss': np.min(losses), 'hparam/nloss': np.min(losses) / rnn_units}
)

writer.close()
