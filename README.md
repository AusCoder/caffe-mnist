## Caffe-mnist

This is a simple example of training a neural network in Caffe with the Mnist dataset. The entire process is run in a docker container, so it should be portable provided you have an nvidia graphics card with CUDA and nvidia-docker installed. This also depends on the versions installed, currently I am running Ubuntu 16.04 with CUDA 8.0.61.

### Lmdb creation

Caffe uses lmdb files to store image and label data for training and testing.

The Dockerfile can be used to create an image that allows you to create lmdb files. You will have to edit the `src/decode.py` file to point to your mnist datafiles. The script can be run in a docker container using docker-compose:
```
docker-compose up lmdb
```

### Training

Once testing and training lmdb files have been created, you can train a neural network. In order to run Caffe training with a GPU in a Docker container, you must have an nvidia graphics card with CUDA and nvidia-docker installed. Installation instructions: (CUDA)[http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf] and (nvidia-docker)[https://github.com/NVIDIA/nvidia-docker].

Once these are installed, you can use the same container as created above to start training. Note that you might have to change the image name.
```
nvidia-docker run --rm -ti \
              -v ~/mnist/data/:/root/mnist/data/ \
              code_lmdb /bin/bash
```

Modulo changes in file paths, you can start training with:
```
caffe train --solver=mnist/data/prototxts/lenet_train_test.prototxt
```
