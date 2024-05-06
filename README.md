# Continual_Learning_with_KANs
Example of continual learning on MNIST dataset using Kolmogorov-Arnold Networks.

For full overview of Kolmogorov-Arnold Networks (KANs) see: https://github.com/KindXiaoming/pykan/ and https://arxiv.org/pdf/2404.19756.

### MNIST
This repository contains a toy example of KANs network used on MNIST dataset to benchmark continual learning capabilities.
MNIST data is permuted according to following rules for different variants (continual learning tasks):
```python
      data, target = data.to(device), target.to(device) # unchanged MNIST dataset
      if variant==1: # first task with permutated MNIST
            data = torch.cat((data[:, :, 14:], data[:, :, :14]), dim=2)
            data = torch.cat((data[:, :, :, 14:], data[:, :, :, :14]), dim=3)
      if variant==2: # second task with permutated MNIST
            data = torch.cat((data[:, :, 14:], data[:, :, :14]), dim=2)
            data = torch.cat((data[:, :, :, 14:], data[:, :, :, :14]), dim=3)
            data = torch.cat((data[:, :, 8:], data[:, :, :8]), dim=2)
            data = torch.cat((data[:, :, :, 8:], data[:, :, :, :8]), dim=3)
```

#### Architecture
Input MNIST image is flatten and passed through only KANs model.
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kan1 = KA(layer_width=[14*14, 128, 10], grid_number=10, k=3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.avg_pool2d(x, 2) # used to reduce the initial dimensions
        x = self.kan1(x)
        output = F.log_softmax(x, dim=1)
        return output
  ```

#### Benchmark
Only test data from MNIST was used to only understand memorization capabilities of the network.

Network was trained for 20 epochs without any modification of the MNIST dataset, then it was switched to `variant=1` task and performance on non-modified MNIST was reported after 10 training epochs.
```python
    for epoch in range(1, 21):
        run(args, model, device, train_loader, optimizer, epoch, test=False, variant=0)

        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=0)
        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=1)

    print("Training new variant")
    for epoch in range(1, 11):
        run(args, model, device, train_loader, optimizer, epoch, test=False, variant=1)

        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=0)
        run(args, model, device, test_loader, optimizer, epoch, test=True, variant=1)
```

#### Results
Different model configuration were tested and the accuracy is reported at the end of first task.

| model configuration KAN parameters | Other paramers | Final Accuracy  | Observation/Comment |
|---|---|---|---|
|  **Learning rate**   |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=**0.001** batch=4  | 0.648  |   |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=**0.0001** batch=4  | **0.903**  |  Lower learning rate is better - **most optimal lr**  |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=**0.00001** batch=4  | 0.715  |  20 epoch on initial training were not sufficient with so low lr  |
|  **Batch size**   |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=**1**  | 0.903  |  No impact with when train with low batch size  |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=**8**  | **0.901**  |  No impact with when train with higher batch size, batch size of 8 will be used for faster training  |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=**16**  | 0.887  |  Higher batch size reduces network performance, probably due to less training steps  |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=**32**  | 0.854  |  Higher batch size reduces network performance, probably due to less training steps  |
|   **Layers width**   |
| layer_width=[14*14, **128**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | **0.901**  |  |
| layer_width=[14*14, **256**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.879  |    |
| layer_width=[14*14, **32**, 10],  grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.758  |    |
| layer_width=[14*14, **32, 16**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.714  |    |
| layer_width=[14*14**, **10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.846  |    |
|   **Grid**   |
| layer_width=[14*14, 128, 10], grid_number=**20**, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.831  |    |
| layer_width=[14*14, 128, 10], grid_number=**10**, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | **0.901**  |    |
| layer_width=[14*14, 128, 10], grid_number=**3**, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.856  |    |
|   **Noise scale**   |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2] noise_scale=**0.0**  | lr=0.0001 batch=8  | 0.148  | Noise_scale is needed |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2] noise_scale=**0.01**  | lr=0.0001 batch=8  | **0.901**  | Default  |
| layer_width=[14*14, 128, 10], grid_number=10, k=3, grid_range=[-2, 2] noise_scale=**0.1**  | lr=0.0001 batch=8  | 0.825  |  Default value seems to work the best  |
|   **Network input size**   |
| layer_width=[**28*28**, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.660  |  |
| layer_width=[**14*14**, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.901  |  |
| layer_width=[**9*9**, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | **0.932**  | Smaller input dimensions improve accuracy on MNIST |
| layer_width=[**7*7**, 128, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.931  |  |
|   **Network hidden size**   |
| layer_width=[9*9, **128**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.932 | |
| layer_width=[9*9, **256**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.967 | |
| layer_width=[9*9, **512**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.975 | |
| layer_width=[9*9, **256, 64**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.947 | |
| layer_width=[9*9, **256, 256, 64**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | 0.937 | |
| layer_width=[9*9, **256, 256**, 10], grid_number=10, k=3, grid_range=[-2, 2]  | lr=0.0001 batch=8  | **0.98** | |
|   **Base function** |
| layer_width=[9*9, 256, 256, 10], grid_number=10, k=3, grid_range=[-2, 2], base_fun=torch.nn.**SiLU**  | lr=0.0001 batch=8  | 0.98 | Default |
| layer_width=[9*9, 256, 256, 10], grid_number=10, k=3, grid_range=[-2, 2], base_fun=torch.nn.**ReLU**  | lr=0.0001 batch=8  | 0.96 | |
| layer_width=[9*9, 256, 256, 10], grid_number=10, k=3, grid_range=[-2, 2], base_fun=torch.nn.**Tanh**  | lr=0.0001 batch=8  | 0.941 | |
| layer_width=[9*9, 256, 256, 10], grid_number=10, k=3, grid_range=[-2, 2], base_fun=torch.nn.**Mish**  | lr=0.0001 batch=8  | **0.986** | |
