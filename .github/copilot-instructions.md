# Reinforcement Learning GridWorld Assignment

This is a Q-learning implementation for a GridWorld environment (EEE 598 Assignment 1). The project explores how different trap penalties affect learned policies in a stochastic environment.

## Architecture Overview

The codebase consists of two main classes in `rlr_assignment_1.py`:

- **GridWorld**: Environment class that defines a 3x4 grid with:

  - Start state at (2,0), goal at (0,3), trap at (1,3)
  - Blocked cell at (1,1) marked with 'X'
  - Stochastic transitions: 80% intended action, 10% left turn, 10% right turn
  - Configurable trap penalty (key experimental parameter)

- **QLearningAgent**: Implements Q-learning with:
  - Epsilon-greedy exploration with decay
  - Live visualization using matplotlib during training
  - Policy extraction showing optimal actions per state

## Key Experimental Setup

The main experiment compares two trap penalty configurations:

- **Trap penalty -1**: Results in policy that goes near trap (see `policy__-1.txt.txt`)
- **Trap penalty -200**: Results in conservative policy avoiding trap (see `policy__-200.txt.txt`)

## Running Experiments

To reproduce results:

```python
# Current configuration uses trap_penalty=-200
env = GridWorld(trap_penalty=-200)  # Change to -1 for comparison
agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=1.0)
agent.visualize_learning_matplotlib(episodes=20_000, epsilon_decay=(1/20_000), min_epsilon=0.01, interval=100)
```

## Output Files Pattern

The codebase generates consistent outputs:

- Policy matrices showing optimal actions: 'N'/'E'/'S'/'W' for movement, 'T' for terminal, 'X' for blocked
- Q-value arrays with shape (height, width, 4) representing action values
- Visualization images saved in `images/` directory with naming pattern: `policy_reward_{penalty}.png`

## Development Notes

- The visualization updates every 100 episodes during training for performance
- State coordinates use (row, col) indexing where (0,0) is top-left
- Actions are indexed as: 0=North, 1=East, 2=South, 3=West
- Living cost of -0.04 per step is hardcoded in the learning loop
- Random seed is not set, so results may vary between runs

## Debugging Q-Learning Issues

When modifying the algorithm:

- Check that `env.step()` vs `env.next_state()` methods are used consistently
- Verify epsilon decay rate matches episode count for proper exploration
- Ensure reward structure includes both immediate rewards and living costs
- Use the live visualization to spot convergence issues or oscillating policies
