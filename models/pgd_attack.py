from ipywidgets import IntProgress
from IPython.display import display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def test_pgd(model, test_loader, epsilon=8 / 256, device='cuda'):
    orig_acc = 0
    rob_acc = 0
    progress = IntProgress(min=0, max=len(test_loader), value=0)
    display(progress)
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        delta = pgd_linf(model, X, y, epsilon)
        yp = model(torch.clamp(X + delta, min=0, max=1))
        y_o = model(X)
        orig_acc += (y_o.max(dim=1)[1] == y).sum()
        rob_acc += (yp.max(dim=1)[1] == y).sum()
        progress.value += 1
    return orig_acc / (len(test_loader) * 128), rob_acc / (len(test_loader) * 128)


def pgd_linf(model, X, y, epsilon=8 / 256, alpha=0.001, num_iter=10, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def plot_pertrubs(rob_accs, pertrub_limit ):
    print(list(map(lambda x: x.item(), rob_accs)))
    #list(map(lambda x: str(x), pertrub_limit))
    plt.plot(range(len(pertrub_limit)), list(map(lambda x: x.item(), rob_accs)))
    plt.xticks(range(len(pertrub_limit)), pertrub_limit)
