from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from minimumkan.KAN import KA

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kan1 = KA(layer_width=[7*7, 256, 256, 10], grid_number=10, k=3, grid_range=[-2, 2], bias_trainable=False, base_fun=torch.nn.Mish())

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.kan1(x)
        output = F.log_softmax(x, dim=1)
        return output


def run(args, model, device, data_loader, optimizer, epoch, test=False, variant=0, vae_train=False, vae_optimizer=None):
    if not test:
        model.train()
    else:
        model.eval()

    correct = 0
    loss_nll = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        if variant==1:
            data = torch.cat((data[:, :, 14:], data[:, :, :14]), dim=2)
            data = torch.cat((data[:, :, :, 14:], data[:, :, :, :14]), dim=3)
        if variant==2:
            data = torch.cat((data[:, :, 14:], data[:, :, :14]), dim=2)
            data = torch.cat((data[:, :, :, 14:], data[:, :, :, :14]), dim=3)
            data = torch.cat((data[:, :, 8:], data[:, :, :8]), dim=2)
            data = torch.cat((data[:, :, :, 8:], data[:, :, :, :8]), dim=3)

        if not test:
            optimizer.zero_grad()
        
        output = model(data)

        # calculate loss
        loss = F.nll_loss(output, target)

        # store losses
        loss_nll += loss.detach()

        # calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if not test:
            loss.backward()
            optimizer.step()
        if args.dry_run:
            break

    print('[{}] {}Mode: {}, Epoch: {}, Loss NLL: {:.4f}, Accuracy: {:.3f}'.format(
        variant,
        "\t" if test else "",
        "test" if test else "train",
        epoch, 
        loss_nll / batch_idx,
        correct / len(data_loader.dataset)
    ))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=False, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        run(args, model, device, train_loader, optimizer, epoch, test=False, variant=0)

        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=0)
        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=1)

    print("Training new variant")
    for epoch in range(1, args.epochs + 1):
        run(args, model, device, train_loader, optimizer, epoch, test=False, variant=1)

        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=0)
        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=1)

    print("Training new variant")
    for epoch in range(1, args.epochs + 1):
        run(args, model, device, train_loader, optimizer, epoch, test=False, variant=2)

        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=0)
        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=1)
        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=2)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
