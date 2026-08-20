# prospect_theory
# Behavioral Reinforcement Learning: Prospect Theory in Uncertain Environments

A reinforcement learning framework bridging behavioral economics and artificial intelligence. This project embeds Daniel Kahneman and Amos Tversky's **Cumulative Prospect Theory (1992)** into tabular Q-learning agents to examine how subjective risk perception, reference dependence, and loss aversion alter learning dynamics under environmental uncertainty.

---

## Motivation & Origin

> *"Man is a deterministic device thrown into a probabilistic universe."* — Michael Lewis, *The Undoing Project*

After reading Michael Lewis's *The Undoing Project*, which chronicles the groundbreaking collaboration between Daniel Kahneman and Amos Tversky, I became fascinated by how human judgment systematically diverges from classical normative rationality. 

Standard reinforcement learning (RL) assumes an agent should act as a strict expected-value maximizer ($E[R]$), indifferent to variance or asymmetric downside risk as long as the arithmetic average remains positive. This project explores whether warping reward signals through empirical human decision models leads to alternative, safer navigation policies in stochastic environments—asking whether there is a "better" way to learn and decide when navigating an uncertain world.

---

## Theoretical Framework

Standard RL updates state-action values using linear environmental reward signals $r$:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

This framework replaces raw reward $r$ with a subjective value function $V(\Delta r)$ evaluated relative to a baseline reference point $r_0$:

$$\Delta r = r - r_0$$

$$V(\Delta r) = \begin{cases} (\Delta r)^\alpha & \text{if } \Delta r \ge 0 \\ -\lambda (-\Delta r)^\beta & \text{if } \Delta r < 0 \end{cases}$$

### Core Behavioral Parameters
* **Reference Expectation ($r_0 = 0.01$):** Encodes an expected minimum progress per step. Failing to reach the goal delivers a negative deviation ($\Delta r = -0.01$).
* **Loss Aversion ($\lambda = 2.25$):** Losses hurt approximately $2.25\times$ more than equivalent gains feel good.
* **Diminishing Sensitivity ($\alpha = \beta = 0.88$):** Non-linear power-law sensitivity yielding risk aversion in gains and risk seeking in losses.

---

## Architecture & Implementation

* **`ProspectTheoryRewardWrapper`:** A custom `gymnasium.RewardWrapper` that intercepts environment feedback during runtime, computing non-linear subjective utility without modifying underlying transition dynamics.
* **Tabular Q-Learning Engine:** Implements $\epsilon$-greedy exploration with exponential decay, updating action-value estimates across discrete state-action spaces.
* **Dual-Agent Benchmark:** Trains standard expected-value maximizers alongside loss-averse prospect agents under identical seeds and transition noise (`is_slippery=True` on `FrozenLake-v1`).

---

## Project Structure

```text
├── main.py              # Full training pipeline, reward wrapper, and comparative plotting
├── README.md            # Theory, motivation, and documentation
└── requirements.txt     # gymnasium, numpy, matplotlib
