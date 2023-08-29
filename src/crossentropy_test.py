import torch
import torch.nn.functional as F

"""
Cross-entropy expects shapes:
input: [C] or [N, C] or [N, C, d_1]
target: [] or [N] or [N, d_1]

N - num of batches
C - num of classes
"""

# [N, C]
input = torch.tensor([
    # batch0
    [0.89, 0.22, 0.33, 0.19],

    # batch1
    [0.89, 0.22, 0.33, 0.19],

    # batch2
    [0.99, 0.01, 0.02, 0.01],
])
print('input shape', input.shape)

# [N]
target = torch.tensor([
    # batch0
    0,

    # batch1
    0,

    # batch2
    3
])
print('target shape', target.shape)


# loss = F.cross_entropy(input, target, reduction='none')
loss = F.nll_loss(F.log_softmax(input, dim=1), target, reduction='none')
print()
print('loss', loss)
print('mean', loss.mean())
print('loss shape', loss.shape)
