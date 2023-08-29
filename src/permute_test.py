import torch

# [N = 2, SEQ = 3, C = 5]
# each batch item has a sequence
#
a = torch.tensor([
    [
        [1, 2, 3, 4, 5],  # seq0:0
        [6, 7, 8, 9, 10],  # seq0:1
        [11, 12, 13, 14, 15],  # seq0:2
    ],
    [
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30],
    ],
])

# expected [N = 2, C = 5, SEQ = 3]

print(a.shape)

b = a.permute(0, 2, 1)
print(b.shape)

print(b)
