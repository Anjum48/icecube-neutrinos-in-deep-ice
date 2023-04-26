# IceCube - Neutrinos in deep ice

Part of the 8th place solution code for the IceCube - Neutrinos in deep ice competition hosted on Kaggle (April 2023) https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice

The writeup can be found here: https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/403713

# Setup

Edit `src/config.py` to reflect the input and output locations on your machine

# Data preparation

1. Download the competition data to the input path defined in config above
2. Run `python src/preprocessing.py`

This will create one `.pt` file for every event. This requires about 1 TB of disk space and enough [inodes](https://www.stackscale.com/blog/inodes-linux/) on your disk for 130 million files. I found I had to split the files across two disks due to the inode limit. Preprocessing takes ~8 hours using 32 threads.

# Training

To train a single model using a config listed in `conf/model` run:

```
python train.py model=<model_name>  # default model is GraphNet/DynEdge
```

To run the 3 models used for my part of the submission (DynEdge, GPS & GravNet) use the shell script `train.sh`

```
sh train.sh
```

# Inference

The final submission code that was used for inference in a Kaggle notebook is in the `submissions` folder

- `submission.py` - Public Score: 0.982, Private Score: 0.98X
