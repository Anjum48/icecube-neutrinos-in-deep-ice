# Preprocessing

## Raw data

This dataset was really big, and repeatedly doing pandas operations every epoch would have been a waste of CPU cycles (polars does not appear to work with PyTorch data loaders with many workers yet). To address this, I made PyTorch Geometric `Data` objects for each event and saved them as `.pt` files which could be loaded during training. This took about 8 hours to create using 32 threads, and required about 1TB of space.

The issue with this was that that 1TB was across 130+ millon tiny files. A Linux partition has a finite number of “index nodes” or `inodes`, i.e. an index to a certain file. Since these files are so small I ran into my inode limit before I ran out of space on my 2TB drive. As a workaround, I had to spread some of these files across two drives, so heads up for anyone trying to reproduce this method or run my code.

A more efficient way could be to store `.pt` files that have already been pre-batched which would require fewer files, but then you lose the ability to shuffle every epoch which may/may not make a difference with this much data. I didn’t try the sqlite method suggested by the GraphNet team.

## Features

In the context of GNNs, each DOM is considered as a node. Each node was given the following 11 features:

- X: X location from sensor\*geometry.csv / 500
- Y: Y location from sensor_geometry.csv / 500
- Z: Z location from sensor_geometry.csv / 500
- T: (Time from batch\*[n].parquet - 1e4) / 3e3
- Charge: log10(Charge from batch\_[n].parquet) / 3.0
- QE: See below
- Aux: (False = -0.5, True = 0.5)
- Scattering Length: See below
- Distance to previous hit: See below
- Time delta since previous hit: See below
- Scattering flag (False = -0.5, True = 0.5). See below

Many of the normalisation methods were taken from GraphNet as a [starting point] (https://github.com/graphnet-team/graphnet/blob/4df8f396400da3cfca4ff1e0593a0c7d1b5b5195/src/graphnet/models/detector/icecube.py#L64-L69), but I altered the scale for time, since time is a very important feature here.

For events with large numbers of hits, to prevent OOM errors, I sampled 256 hits. This can make the process slightly non-deterministic.
Quantum efficiency
QE is the quantum efficiency of the photomutipliers in the DOMs. The DeepCore DOMs are quoted to have 35% higher QE than the regular DOMs (Figure 1 of this [paper](https://arxiv.org/pdf/2209.03042.pdf)), so QE was set to 1 everywhere, and 1.35 for the lower 50 DOMs in DeepCore. The final QE feature was scaled using (QE - 1.25) / 0.25.
Scattering length
Scattering and absorption lengths are important to characterise differences in the clarity of the ice. This data is published on page 31 of this [paper](https://arxiv.org/abs/1301.5361). A datum depth of 1920 metres was used so that z = (depth - 1920) / 500. The data was resampled to the z values using `scipy.interpolate.interp1d`. I found that after passing the data though `RobustScaler`, the scattering and absorption data was near identical, so I only used scattering length.
Previous hit features
The two main types of events are track and cascade events. Looking at some of the amazing [visualisation tools](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/388858) for example from edguy99, I got the idea that if a node had some understanding where and when the nearest previous hit was, it might help the model differentiate between these two groups. To calculate this for each event, sorted the hits by time, calculated the pairwise distances of all hits, masked any hits from the future and calculated the distance, d, to the nearest previous hit. This was scaled using (d - 0.5) / 0.5. The time delta from the previous hit was also calculated using the same method and scaled using (t - 0.1) / 0.1.
Scattering flag
I tried to create a flag that could discern whether a hit was caused directly from a track, or some secondary scattering, inspired by section 2.1 of this [paper](https://arxiv.org/pdf/2203.02303.pdf). A side effect of adding this flag was that training was much more stable. The flag is generated as follows:
Identify the hit with the largest charge
From this DOM location, calculate the distances & time delta to every other hit
If the time taken to travel that distance is > speed of light in ice, assume that the photon is a result of scattering

# Validation

I used 90% - 10% train-validation split, and no cross validation due to the size of the dataset. The split was done by creating 10 bins of log10(n_hits), and then using `StratifiedKFold` using 10 splits.

# Models

I used the `DirectionReconstructionWithKappa` task directly from GraphNet, meaning that an embedding (e.g. shape of 128) will be projected to a shape of 4 (x, y, z, kappa)

## Architectures

I used the following 3 architectures. All validation scores are with 6x TTA applied

- [GraphNet/DynEdge](https://github.com/graphnet-team/graphnet) - Val = 0.98501
- [GPS](https://arxiv.org/abs/2205.12454) - Val = 0.98945\*
- [GravNet](https://arxiv.org/abs/1902.07987) - Val = 0.98519

The average of these 3 models gave a 0.982 LB score.

The GraphNet/DynEdge model had very little modfification, other than changing to GELU activations.

GPS & GravNet used 8 blocks and you can find the exact architectures for both in the code [here](https://github.com/Anjum48/icecube-neutrinos-in-deep-ice/blob/main/src/modules.py).

\*GPS was the most powerful model, but also slowest to train being a transformer type model (roughly 11 hours/epoch on my machine). I managed to train a model which achieved a validation score of 0.98XX but was too late to include in our final submission.

## Loss

I used VonMisesFisher3DLoss + (1 - CosineSimilarity) as the final loss function, since cosine similarity is a nice proxy for mean angular error. For CosineSimilarity I transformed the target azimuth & zenith values to cartesian coordinates.

This performed much better than separate losses for azimuth (VMF2D) & zenith (MSE) which I the route I [initially went down](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/383546).

## Augmentation

I centered the data on string 35 (the DeepCore string) and rotated about the z-axis in 60 degree steps. This didn’t actually improve validation performance but did have the benefit of making the models rotationally invariant so that I could take advantage of the detector symmetry and apply a 6x test time augmentation (TTA). This often improved scores by 0.002-0.003.

## Training parameters

AdamW optimiser
Epochs = 6
Cosine schedule (no warmup)
Learning rate = 0.0002
Batch size = 1024
Weight decay = 0.001 - 0.1 depending on model
FP16 training
Hardware: 2x RTX 3090, 128 GB RAM

# Final submissions

Circular mean

# Robustness to perturbation

TBC

# Lessons learned/stuff that didn’t work

The GraphNet DynEdge baseline is extremely strong and tough to improve on - kudos to the team! It is also the fastest/efficient model, and what I used for the majority of experimentation
More data = more better. The issue with this though is that I found that some conclusions drawn from experiments on 1% or 5% of the data were no longer applicable on the full dataset. This made experimentation slow and expensive
Batch normalisation made things unstable and didn’t show improvements
Lion optimiser didn’t generalise as well as AdamW
Weight decay was important for some models. As a result I assumed changing the epsilon value in Adam would have an effect, but I didn’t see anything significant
In my experiments, GNNs seem to benefit from leaky activations, e.g. GELU
For MPNN aggregation, it seems that [min, max, mean, sum] is sufficient. Adding more didn’t appear to make significant gains
Realigning all of the times to the time of the first hit of each event deteriorates performance, possibly due to noise in the data/false triggers etc.
Radial nearest neighbours didn’t work any better than KNN when defining graph edges
Only using 1 - CosineSimilarity as a loss function wasn’t very stable. Adding VMF3D helped a lot

# Code

All my code is available here: https://github.com/Anjum48/icecube-neutrinos-in-deep-ice
