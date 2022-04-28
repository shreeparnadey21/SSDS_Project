# SSDS_Project

To run Horovod file on GPU cluster:

First set the parameters with par dictionary:

Example:

par={'nproc':1,'batch_size':8, 'epochs':1, 'cuda':torch.cuda.is_available(), 'model':'resnet50', 'dataset':'MNIST'}

For example to run on 4 GPUs: horovodrun -np 4 python3 SSDS_Horovod.py


To run RayTrain file on GPU cluster:

First set the parameters with par dictionary:

Example:

par={'nproc':2,'batch_size':32, 'epochs':1, 'cuda':torch.cuda.is_available(), 'model':'resnet50', 'dataset':'MNIST'}

To run the file: python3 SSDS_RayTrain.py

References for code:

SSDS_Horovod.py reference: https://github.com/horovod/horovod/tree/master/examples/pytorch

SSDS_RayTrain reference: https://docs.ray.io/en/latest/raysgd/raysgd.html, 
https://docs.ray.io/en/latest/train/examples/train_fashion_mnist_example.html

SSDS_DistPytorch: https://github.com/olehb/pytorch_ddp_tutorial/blob/main/ddp_tutorial_multi_gpu.py
The Training and Testing code used in SSDS_DistPytorch.py is directly taken from the above Github site.
This code itself was not properly running on cluster and due to lack of time we could't make desired modifications to code.
