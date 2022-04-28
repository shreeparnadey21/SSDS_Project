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
https://github.com/horovod/horovod
https://docs.ray.io/en/latest/raysgd/raysgd.html
