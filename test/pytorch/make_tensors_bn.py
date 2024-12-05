import os
import random
import sys
import torch
import torch.nn.functional as F

def main():
    names_fn = sys.argv[1]
    data_dir = sys.argv[2]

    os.makedirs(data_dir, exist_ok=True)

    shapes_fn = f"{data_dir}/shapes.hpp"
    words = open(names_fn).read().splitlines()
    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    block_size = 3 # context length: how many characters do we take to predict the next one?

    def build_dataset(words):  
        X, Y = [], []
        for w in words:
            #print(w)
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y.view(-1, 1)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words)
    Xtr = Xtr[:32]
    Ytr = Ytr[:32]

    g = torch.Generator().manual_seed(2147483647) # for reproducibility

    # Create embedding lookup table C
    # Embed our 27 possible characters into a lower dimensional space
    # Each character will have a 2D embedding
    C = torch.randn((27, 2), generator=g, requires_grad=True)
    emb = C[Xtr]  # shape: 32, 3, 2

    hidden_size = 10

    W1 = torch.randn((3 * 2, hidden_size), generator=g, requires_grad=True)
    W2 = torch.randn((hidden_size, 27), generator=g, requires_grad=True)
    b2 = torch.randn((1, 27), requires_grad=True)
    bn = torch.nn.BatchNorm1d(hidden_size, momentum=0.001, eps=1e-5)
    cl = torch.nn.CrossEntropyLoss()

    emb_view = emb.view(emb.shape[0], W1.shape[0])
    hprebn = emb_view @ W1
    hpreact = bn(hprebn)
    h = torch.tanh(hpreact)

    h_w2 = h @ W2
    logits = h_w2 + b2

    softmax = torch.nn.Softmax(dim=1)
    dlogits = softmax(logits)
    print(dlogits)
    loss = cl(logits, Ytr.squeeze())
    # # cross entropy loss (same as F.cross_entropy(logits, Yb))
    # logit_maxes = logits.max(1, keepdim=True).values
    # norm_logits = logits - logit_maxes # subtract max for numerical stability
    # counts = norm_logits.exp()
    # counts_sum = counts.sum(1, keepdims=True)
    # counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    # probs = counts * counts_sum_inv
    # logprobs = probs.log()
    # loss = -logprobs[range(Ytr.shape[0]), Ytr.squeeze()].mean()

    params = [loss, logits, h_w2, h, hpreact, hprebn, emb_view, C, W1, W2, b2, Xtr, Ytr]
    params_str = ["loss", "logits", "h_w2", "h", "hpreact", "hprebn", "emb_view", "C", "W1", "W2", "b2", "Xtr", "Ytr"]

    for p, n in zip(params, params_str):
        if p.dtype in {torch.float32, torch.float64}:
            p.retain_grad()

    loss.backward()

    with open(shapes_fn, "w") as f:
        f.write("std::unordered_map<std::string, std::array<size_t, 2>> shapes = {\n")
        for j, (p, n) in enumerate(zip(params, params_str)):
            if p.grad is not None:
                p.grad.numpy().tofile(f"{data_dir}/{n}.grad.bin")
            p.detach().numpy().tofile(f"{data_dir}/{n}.bin")

            # Write shape to a header file
            f.write(f"    {{\"{n}\", {{")

            if len(p.shape) == 0:
                f.write("1, 1")
            else:
                for i, s in enumerate(p.shape):
                    f.write(f"{s}")
                    if i < len(p.shape) - 1:
                        f.write(", ")
            f.write("}}")
            if j < len(params) - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n")



if __name__ == "__main__":
    main()