# Isamu Part

Before the team merge, I created LSTM and GraphNet models. After the team merge, I focused on GraphNet because Remek's LSTM model was superior to mine. I used graphnet (https://github.com/graphnet-team/graphnet) as a baseline and made several changes to improve its accuracy. I list below some of the experiments I performed that worked(There are tons of things that didn't work)

- random sampling (random sampling from DB if the specified data length exceeds 800)
- Increasing nearest neighbors of KNN layer(8->16)
- Addition of features
  - x, y, z
  - time
  - charge
  - auxiliary
  - ice_transparency feature
- 2-stage model with kappa(sigma) of vonMisesFisher distribution 
  - Train 1st stage model to predict x, y, z, kappa
      - restart from the weights of GraphNet from public baseline notebook
      - About 1-250 batches were used
      - batch size 512
      - epoch 20
      - DirectionReconstructionWithKappa
  - Split data into easy and hard parts according to 1st stage kappa value
       - Inference was performed using the 1st model and classified into two sets of data(easy part and hard part) according to their predicted kappa value
  - Train expert models for easy and hard parts and combine their predictions
      - About 250-350 batches were used
      - batch size 512
      - epoch 20
      - DirectionReconstructionWithKappa
- TTA 
  - rotation 180-degree TTA about the z-axis
- Loss
  - DirectionReconstructionWithKappa
- Hardware: RTX 3090, 64 GB RAM, 8TB HDD, 2TB SSD, Google Colab Pro

The ensemble of models(1st and 2nd) created above gave public LB 0.995669, private LB 0.996550 
