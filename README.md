## TAdam

The Pytorch implementation of TSGD algorithm in：'Scaling transition from SGDM to plain SGD'
[https://arxiv.org/abs/2106.06749](https://arxiv.org/abs/2106.06753)

### Usage

```python
from tsgd import TSGD

...

optimizer = TSGD(model.parameters(), iters=required, momentum=0.9, lr=1e-3, moment=3/8, up_lr=0.1, low_lr=0.005)

#iters(int, required): iterations
#	iters = (testSampleSize / batchSize) * epoch
#
#moment(float, optional): transition moment
#       moment = transition_iters / iters

#set default value: momentum=0.9, moment=3/8, up_lr=0.1, low_lr=0.005
```





The code will be uploaded as soon as possible

