# [AutoToM](https://chuanyangjin.com/AutoToM/) Experiment 3: Embodied Assistance

## Findings

 [Project Page](https://chuanyangjin.com/AutoToM/#:~:text=Experiment%203%3A%20Embodied%20Assistance) | [Paper Section](https://arxiv.org/pdf/2502.15676v2#page=11.55) | [Details Section](https://arxiv.org/pdf/2502.15676v2#page=28.75)

We evaluated AutoToM in an embodied assistance benchmark, [Online Watch-And-Help (O-WAH)](https://www.tshu.io/online_watch_and_help/), where a helper agent must simultaneously observe a main agent's actions, infer its goal, and assist it to reach the inferred goal faster in realistic household environments.

<p align="center">
  <img src="https://chuanyangjin.com/AutoToM/static/img/experiment3.png" width="50%">
  <br>
  <em>Averaged speedup of AutoToM and baselines on the O-WAH benchmark.</em>
</p>

- Random Goal baseline achieves a 6.3% speedup, but with high variance and negative speedup in 50% of the episodes;
- GPT-4o achieves a similar but more stable speedup of 6.8%;
- In contrast, AutoToM achieves the highest speedup of 27.7%, significantly outperforming all baselines.

This is because AutoToM can produce more accurate uncertainty estimation of goal hypotheses based on observed actions, which is key to generating robust and useful helping plans.

## Documentation
- Setup: [docs/setup.md](/docs/setup.md)
- Usage: [docs/usage.md](/docs/usage.md)
- Code Structure: [.cursor/rules/code-structure.mdc](/.cursor/rules/code-structure.mdc)
