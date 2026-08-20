import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

#cite AI + the undoing project 
def train_q_learning(
    env_name="FrozenLake-v1",
    episodes=2000,
    alpha=0.1,  # Learning rate
    gamma=0.99,  # Discount factor
    epsilon=1.0,  # Starting exploration rate
    epsilon_decay=0.995,  # Decay rate per episode
    min_epsilon=0.01,
):
    """ Trains a Q-learning agent to navigate a Frozen Lake grid through trial and error.

    The objective is to navigate to the treasure chest while avoiding ice holes.
    The agent maintains a table of scores (Q-table) for moving Up, Down, Left, or Right
    from each tile.

    Initially, all scores are 0. The agent moves at random to explore the map,
    gradually learning to select optimal moves based on the highest accumulated scores.

    Parameters:
        env_name (str): Gymnasium environment name.
        episodes (int): Total training runs.
        alpha (float): Learning rate (how fast new feedback overwrites old scores).
        gamma (float): Discount factor for future rewards.
        epsilon (float): Starting probability of taking a random exploratory move.
        epsilon_decay (float): Multiplicative decay rate for epsilon per episode.
        min_epsilon (float): Minimum threshold for random exploration.

    Returns:
        tuple: (q_table, episode_returns) trained Q-table and list of total returns per episode."""
    # 1. Initialize environment (is_slippery=True introduces transition uncertainty)
    env = gym.make(env_name, is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # 2. Initialize Q-table with zeros
    q_table = np.zeros((n_states, n_actions))

    # Tracking metrics
    episode_returns = []

    print(f"--- Training Tabular Q-Learning Agent on {env_name} ---")

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Standard Q-learning Update Rule:
            # Q(s,a) <-- Q(s,a) + alpha * [r + gamma * max_a Q(s',a) - Q(s,a)]
            best_next_action = np.argmax(q_table[next_state, :])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (
                1 - done
            )
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_returns.append(total_reward)

        if (episode + 1) % 500 == 0:
            avg_return = np.mean(episode_returns[-100:])
            print(
                f"Episode {episode + 1}/{episodes} | Avg Return (Last 100): {avg_return:.2f} | Epsilon: {epsilon:.3f}"
            )

    env.close()
    return q_table, episode_returns


def plot_returns(returns, window=50):
    """Plots raw episode returns alongside a rolling average curve."""
    plt.figure(figsize=(10, 5))

    # Rolling average calculation
    smoothed_returns = np.convolve(
        returns, np.ones(window) / window, mode="valid"
    )

    plt.plot(returns, alpha=0.3, color="gray", label="Raw Episode Return")
    plt.plot(
        range(window - 1, len(returns)),
        smoothed_returns,
        color="blue",
        linewidth=2,
        label=f"{window}-Episode Moving Average",
    )

    plt.title("Baseline Q-Learning Agent Performance (FrozenLake-v1)")
    plt.xlabel("Episode")
    plt.ylabel("Raw Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train the agent and plot baseline performance
    q_table, returns = train_q_learning(episodes=2000)
    plot_returns(returns)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class ProspectTheoryRewardWrapper(gym.RewardWrapper):
    """Transforms raw environment rewards into subjective utility using

    Kahneman & Tversky's (1992) Cumulative Prospect Theory.

    Instead of treating rewards as linear expected values, this wrapper
    distorts rewards relative to a reference point (r0). It applies loss
    aversion (losses hurt ~2.25x more than equivalent gains feel good)
    and diminishing sensitivity (power law) to simulate human risk perception.

    Mathematical Formulation:
      delta_r = reward - r0
      V(delta_r) = (delta_r)^alpha                 if delta_r >= 0  (Gains)
      V(delta_r) = -lambda_loss * (-delta_r)^beta  if delta_r < 0   (Losses)

    Parameters:
        env (gym.Env): The Gymnasium environment to wrap.
        r0 (float): Baseline expectation/reference point (default: 0.01).
        alpha (float): Gain concavity exponent (default: 0.88).
        beta (float): Loss convexity exponent (default: 0.88).
        lambda_loss (float): Loss aversion coefficient (default: 2.25).
    """

    def __init__(
        self, env, r0: float = 0.01, alpha=0.88, beta=0.88, lambda_loss=2.25
    ):
        super().__init__(env)
        self.r0 = r0
        self.alpha = alpha
        self.beta = beta
        self.lambda_loss = lambda_loss

    def reward(self, reward: float) -> float:
        # 1. Deviations from reference point (r0)
        delta_r = reward - self.r0

        if delta_r >= 0:
            # Gain regime: Concave utility (diminishing sensitivity)
            return float(delta_r**self.alpha)
        else:
            # Loss regime: Convex utility penalized by lambda_loss (2.25x)
            return float(-self.lambda_loss * ((-delta_r) ** self.beta))


def train_q_learning(
    env,
    episodes=5000,  # Increase episodes so it has time to propagate rewards
    alpha=0.2,  # Slightly higher learning rate (0.2 instead of 0.1)
    gamma=0.95,  # 0.95 focuses slightly more on realistic step horizons
    epsilon=1.0,
    epsilon_decay=0.999,  # Slower decay (stays exploring longer through ~3000 episodes)
    min_epsilon=0.05,  # Keep 5% random exploration active
):
    """Trains a tabular Q-learning agent on a given Gymnasium environment.

    Parameters:
        env (gym.Env): Instantiated environment (standard or wrapped).
        episodes (int): Total training runs.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Starting probability of random exploration.
        epsilon_decay (float): Multiplicative decay per episode.
        min_epsilon (float): Floor for exploration.

    Returns:
        tuple: (q_table, episode_returns)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions))
    episode_returns = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Standard Q-learning Update Rule:
            # Q(s,a) <-- Q(s,a) + alpha * [r + gamma * max_a Q(s',a) - Q(s,a)]
            best_next_action = np.argmax(q_table[next_state, :])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (
                1 - done
            )
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_returns.append(total_reward)

        if (episode + 1) % 500 == 0:
            avg_return = np.mean(episode_returns[-100:])
            print(
                f"Episode {episode + 1}/{episodes} | Avg Return (Last 100): {avg_return:.3f} | Epsilon: {epsilon:.3f}"
            )

    env.close()
    return q_table, episode_returns


def plot_comparison(standard_returns, prospect_returns, window: int = 50):
    """Plots a smoothed rolling comparison of standard vs.

    prospect RL agents.
    """
    plt.figure(figsize=(11, 5.5))

    std_smoothed = np.convolve(
        standard_returns, np.ones(window) / window, mode="valid"
    )
    prospect_smoothed = np.convolve(
        prospect_returns, np.ones(window) / window, mode="valid"
    )

    x_range = range(window - 1, len(standard_returns))

    plt.plot(
        x_range,
        std_smoothed,
        color="steelblue",
        linewidth=2,
        label="Standard Agent ($E[R]$ Maximizer)",
    )
    plt.plot(
        x_range,
        prospect_smoothed,
        color="crimson",
        linewidth=2,
        label="Prospect Theory Agent (Loss Averse, $\\lambda=2.25$)",
    )

    plt.title(
        "Standard vs. Prospect Theory Q-Learning on FrozenLake-v1", fontsize=13
    )
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel(f"{window}-Episode Moving Average Return", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    TOTAL_EPISODES = 2500

    # 1. Baseline Standard Agent
    print("=== Training Standard Q-Learning Agent ===")
    raw_env = gym.make("FrozenLake-v1", is_slippery=True)
    q_table_std, std_returns = train_q_learning(
        raw_env, episodes=TOTAL_EPISODES
    )

    # 2. Prospect Theory Agent
    print("\n=== Training Prospect Theory Agent ===")
    base_env = gym.make("FrozenLake-v1", is_slippery=True)
    # r0=0.01 creates slight step penalty (time cost), punishing meandering near holes
    prospect_env = ProspectTheoryRewardWrapper(
        base_env, r0=0.01, lambda_loss=2.25
    )
    q_table_prospect, prospect_returns = train_q_learning(
        prospect_env, episodes=TOTAL_EPISODES
    )

    # 3. Compare Results
    plot_comparison(std_returns, prospect_returns)

if __name__ == "__main__":
    TOTAL_EPISODES = 2000

    # 1. Train Standard Agent
    print("=== 1/2 Training Standard Expected Value Agent ===")
    raw_env = gym.make("FrozenLake-v1", is_slippery=True)
    _, std_returns = train_q_learning(raw_env, episodes=TOTAL_EPISODES)

    # 2. Train Prospect Theory Agent
    print("\n=== 2/2 Training Prospect Theory Agent ===")
    base_env = gym.make("FrozenLake-v1", is_slippery=True)
    prospect_env = ProspectTheoryRewardWrapper(
        base_env, r0=0.01, lambda_loss=2.25
    )
    _, prospect_returns = train_q_learning(prospect_env, episodes=TOTAL_EPISODES)

    # 3. Plot Both Side-by-Side
    plot_comparison(std_returns, prospect_returns)