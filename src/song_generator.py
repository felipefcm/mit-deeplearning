import torch

import song_generator.vocab as songvocab
from song_generator.network import SongNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'

rnn_units = 1024
embedding_dim = 256

net = SongNetwork(
    rnn_units=rnn_units,
    vocab_size=songvocab.vocab_size,
    embedding_dim=embedding_dim
)

net.eval()
net = net.to(device)
# net.load_state_dict(torch.load('checkpoints/lr0.005_bs64_sl900_rnn1300'))
net.load_state_dict(torch.load('checkpoints/lr0.005_bs64_sl500_rnn1024'))
print('loaded checkpoint')

seed = 'T:The Turtle\n'

for i in range(300):
    encoded = songvocab.vectorise_string(seed)

    input = torch.tensor([encoded]).to(device)

    with torch.no_grad():
        predicted = net(input)

    chars = songvocab.take_predicted_sample(predicted[0])
    seed += chars[-1]

print('song', seed)
