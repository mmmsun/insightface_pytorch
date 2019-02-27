# FaceFlamingo

Pytorch implementation of face recognition

![FaceFlamingo](fig/giphy.gif)


# Requirements

- python>=3.6
- pytorch
- ignite
- pip install --upgrade git+https://github.com/tensorpack/tensorpack.git


# Usage

Start tensorboard:

```bash
tensorboard --logdir=/tmp/tensorboard_logs/
```

Run the example:

```bash
python train_face.py $CONFIG_PATH
```

# Combinations

- arch (backbone)
- embedding layer
- loss