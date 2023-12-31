import numpy as np
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import song_generator.songdataset as songdataset
import song_generator.vocab as songvocab
from song_generator.network import SongNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'

should_log = False

rnn_units = 1024
embedding_dim = 256
lr = 0.005
num_epochs = 1000
seq_length = 500
batch_size = 64

writer = None

if should_log:
    writer = SummaryWriter(
        log_dir=f'runs/lr{lr}_bs{batch_size}_sl{seq_length}_rnn{rnn_units}'
    )

net = SongNetwork(
    rnn_units=rnn_units,
    vocab_size=songvocab.vocab_size,
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
        songvocab.vectorised_songs,
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

    if writer:
        writer.add_scalar('Loss/mean', loss.item(), epoch)

    losses.append(loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    lr_scheduler.step()

    if epoch % 100 == 0 or epoch == num_epochs - 1:
        net.eval()
        with torch.no_grad():
            chars = songvocab.take_predicted_sample(output[0])
            if writer:
                writer.add_text(
                    'Predicted', f'loss={loss.item()}\n{chars}', epoch
                )

    if (epoch + 1) % 100 == 0:
        time.sleep(2)

if writer:
    writer.add_hparams(
        hparam_dict,
        {'hparam/loss': np.min(losses),
         'hparam/nloss': np.min(losses) / rnn_units}
    )

torch.save(
    net.state_dict(),
    f'checkpoints/lr{lr}_bs{batch_size}_sl{seq_length}_rnn{rnn_units}'
)

if writer:
    writer.close()
