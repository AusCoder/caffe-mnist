#!/bin/bash

nvidia-docker run --rm -ti \
              -v ~/mnist/data/:/root/mnist/data/ \
              code_lmdb /bin/bash
