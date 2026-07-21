import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


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
    