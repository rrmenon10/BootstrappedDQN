# Bootstrapped DQN

This repository maintains the code for Bootstrapped DQN (Osband 2016).

Experiments are being run to check the best suited version (DDQN/DQN, with/without gradient normalisation, number of heads) and the plots will be put up here soon :)

The current implementation uses DQN with Bootstrap architecture.

## Preliminary Graphs

### Breakout 2 Heads

The current implementation involves using 2 heads without gradient normalisation for 100 epochs.

![average_reward_100epoch](https://cloud.githubusercontent.com/assets/8466046/22464239/e2113e22-e7db-11e6-9275-974b7770aa6c.png)

### Breakout 10 heads (with and w/o gradient normalization)

This is the first result with 10 heads with BootstrappedDQN.

![average_reward](https://cloud.githubusercontent.com/assets/8466046/22435011/60353422-e744-11e6-87a7-dc08ad2142a2.png)
