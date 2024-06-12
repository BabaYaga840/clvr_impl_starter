# Design choices


For the reconstructions from encoders trained on horizontal and verticle rewards, used: 
```
-a smaller bottleneck  
-a weighted loss function: Higher emphasis on reconstructing white pixels (indicated as ww in code)
```
RL algorithm used: PPO

Distribution used to sample actions: Beta distribution

# Observations


-Rewards wise ORACLE > Encoder finetune > Encoder pretrain > Image ~ RANDOM POLICY

-Initial rewards in every run is extremely variable (lies between 26-28)

-Using a beta distribution to model the actions seemed to work better compared to a normal distribution
