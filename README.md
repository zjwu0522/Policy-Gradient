# Policy-Gradient
This is a pytorch version implementation for the paper:

>Sutton, Richard S., et al. ["Policy gradient methods for reinforcement learning with function approximation."](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) Advances in neural information processing systems. 2000.

and

>Schulman, John, et al. ["Proximal policy optimization algorithms."](https://arxiv.org/abs/1707.06347) arXiv preprint arXiv:1707.06347 (2017).

## Code-level Optimizations
Many people find reinforcement learning methods are hard to reproduce. A paper by MIT and Two Sigma in ICLR2020 investigated the consequences of code-level optimizations, which are known as algorithm augmentations found only in implementations or simply, "tricks", to the core algorithm:

>Engstrom, Logan, et al. ["Implementation matters in deep policy gradients: A case study on ppo and trpo."](https://arxiv.org/pdf/2005.12729.pdf) arXiv preprint arXiv:2005.12729 (2020).

For those who want to implement their own policy gradient methods, refer to this paper and to find some useful insights towards the difiiculty and importance of attributing performance gains in deep reinforcement learning.

## Useful Links
1. [PPO Expalained](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)
2. [Policy Gradient and PPO Notebook (Chinese)](https://xiang578.com/post/reinforce-learnning-basic.html)
3. [如何直观理解PPO算法? (Chinese)](https://zhuanlan.zhihu.com/p/111049450)
