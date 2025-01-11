import torch


if __name__ == '__main__':
    x = torch.load('/tmp/x.pt')
    w = torch.load('/tmp/w.pt')


    for _ in range(128):
        y = w.forward(x)
        y = y.squeeze()
        print(y.flatten()[0])
