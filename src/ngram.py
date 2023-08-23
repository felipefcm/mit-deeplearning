import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".lower().split()

vocab = sorted(set(test_sentence))
word2idx = {w: i for i, w in enumerate(vocab)}

context_size = 2
embedding_dim = 10

ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(context_size)],
        test_sentence[i]
    )
    for i in range(context_size, len(test_sentence))
]


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.l1 = nn.Linear(context_size * embedding_dim, 128)
        self.l2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        embeds = self.emb(x).view(1, -1)

        out = F.relu(self.l1(embeds))
        out = self.l2(out)

        log_probs = F.log_softmax(out, dim=1)
        return log_probs


torch.manual_seed(1)

losses = []
loss_fn = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab))
optim = torch.optim.SGD(model.parameters(), lr=0.001)

dataset = TensorDataset()

for epoch in range(1000):
    total_loss = 0
    for context, target in ngrams:
        context_idx = torch.tensor([word2idx[w] for w in context])

        log_probs = model(context_idx)
        loss = loss_fn(log_probs, torch.tensor([word2idx[target]]))

        loss.backward()
        optim.step()
        model.zero_grad()

        total_loss += loss.item()

    losses.append(total_loss)

print('losses', losses[0], losses[-1])

# print('beauty', model.emb.weight[word2idx['beauty']])
# print('thy', model.emb.weight[word2idx['thy']])
# print('shall', model.emb.weight[word2idx['shall']])

print('cos', F.cosine_similarity(
    model.emb.weight[word2idx['beauty']], model.emb.weight[word2idx['thy']], dim=0))

print('cos2', F.cosine_similarity(
    model.emb.weight[word2idx['beauty']], model.emb.weight[word2idx['praise']], dim=0))
