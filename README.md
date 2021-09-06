## TSGD

The Pytorch implementation of TSGD algorithm in：'Scaling transition from SGDM to plain SGD'
[https://arxiv.org/abs/2106.06749](https://arxiv.org/abs/2106.06753)  
The implementation is highly based on projects [AdaBound](https://github.com/Luolc/AdaBound) , [Adam](https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/adam.py) , [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), thanks pretty work.  
The test environment we passed is: PyTorch=1.7.0, Python=3.7.10, Linux/Centos8.3.  

### Usage

Please directly download the [TSGD](https://github.com/kunzeng/TSGD/tree/main/tsgd) folder and put it in your project, then

```python
from tsgd import TSGD

...

optimizer = TSGD(model.parameters(), iters=required)

#iters(int, required): iterations
#	iters = (testSampleSize / batchSize) * epoch
```



The code will be uploaded as soon as possible.

