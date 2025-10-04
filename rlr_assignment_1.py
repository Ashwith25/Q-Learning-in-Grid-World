import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from matplotlib.gridspec import GridSpec

class GridWorld:
    def __init__(self, height=3, width=4, block=[(1,1)], start=(2,0), goal=(0,3), trap=(1,3), trap_penalty=-1):
        self.height = height
        self.width = width
        self.block = block
        self.start = start
        self.actions = ['N', 'E', 'S', 'W']
        self.rewards = np.zeros((height, width))
        self.rewards[goal] = 1
        self.rewards[trap] = trap_penalty
        self.terminal_states = [goal, trap]

    def is_terminal(self, state):
        return (state) in self.terminal_states

    def step(self, state, action):
        if self.is_terminal(state):
            return state
        transitions = self._get_transition_probs(action)
        next_states = []
        probs = []
        for a, p in transitions.items():
            ns = self._move(state, a)
            next_states.append(ns)
            probs.append(p)
        idx = random.choices(range(len(next_states)), weights=probs)[0]
        return next_states[idx]

    def _get_transition_probs(self, action):
        # 80% intended, 10% left, 10% right
        idx = self.actions.index(action)
        left = self.actions[(idx - 1) % 4]
        right = self.actions[(idx + 1) % 4]
        return {action: 0.8, left: 0.1, right: 0.1}

    def _move(self, state, action):
        i, j = state
        if action == 'N': i -= 1
        elif action == 'E': j += 1
        elif action == 'S': i += 1
        elif action == 'W': j -= 1
        if (i < 0 or i >= self.height or j < 0 or j >= self.width or (i,j) in self.block):
            return state
        return (i, j)

class QLearningAgent:


    def __init__(self, env, alpha=0.1, gamma=0.1, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.height, env.width, len(env.actions)))
        
        # Tracking for plots
        self.episode_rewards = []
        self.epsilon_history = []
        self.episode_numbers = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        i, j = state
        return self.env.actions[np.argmax(self.Q[i, j])]

    def learn_with_tracking(self, episodes=1000, epsilon_decay_rate=None, min_epsilon=0.01):
        """Learn with episode reward and epsilon tracking for plotting"""
        if epsilon_decay_rate is None:
            epsilon_decay_rate = 0.995  # Default decay rate if not provided
            
        for ep in range(episodes):
            state = self.env.start
            episode_reward = 0
            
            while not self.env.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.env.step(state, action)  # Use step() for consistency
                i, j = state
                ni, nj = next_state
                a_idx = self.env.actions.index(action)
                
                reward = -0.04 + self.env.rewards[next_state]
                episode_reward += reward
                
                best_next = np.max(self.Q[ni, nj])
                self.Q[i, j, a_idx] += self.alpha * (reward + self.gamma * best_next - self.Q[i, j, a_idx])
                state = next_state
            
            self.episode_rewards.append(episode_reward)
            self.epsilon_history.append(self.epsilon)
            self.episode_numbers.append(ep + 1)
            
            self.epsilon = max(self.epsilon * epsilon_decay_rate, min_epsilon)
    
    def learn(self, episodes=1000, epsilon_decay=0.995, min_epsilon=0.01):
        for ep in range(episodes):
            state = self.env.start
            while not self.env.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.env.step(state, action)  # Fixed: use step() instead of next_state()
                i, j = state
                ni, nj = next_state
                a_idx = self.env.actions.index(action)
                reward = self.env.rewards[next_state]
                best_next = np.max(self.Q[ni, nj])
                self.Q[i, j, a_idx] += self.alpha * (reward + self.gamma * best_next - self.Q[i, j, a_idx])
                state = next_state
            
            self.epsilon = max(self.epsilon * epsilon_decay, min_epsilon)

    def visualize_learning_matplotlib(self, episodes=1000, epsilon_decay=0.995, min_epsilon=0.01, interval=1):
        fig, ax = plt.subplots(figsize=(self.env.width, self.env.height))
        def draw_grid(Q):
            ax.clear()
            ax.set_xticks(np.arange(-0.5, self.env.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.env.height, 1), minor=True)
            ax.grid(which='minor', color='black', linewidth=2)
            ax.set_xlim(-0.5, self.env.width-0.5)
            ax.set_ylim(self.env.height-0.5, -0.5)
            for i in range(self.env.height):
                for j in range(self.env.width):
                    if (i, j) in self.env.block:
                        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='gray'))
                        ax.text(j, i, 'X', ha='center', va='center', fontsize=16, color='white')
                    elif (i, j) in self.env.terminal_states:
                        val = self.env.rewards[i, j]
                        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='lightgreen' if val > 0 else 'red'))
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=16)
                    else:
                        qvals = Q[i, j]
                        txt = '\n'.join([f'{a}:{qvals[k]:.2f}' for k, a in enumerate(self.env.actions)])
                        ax.text(j, i, txt, ha='center', va='center', fontsize=8)
                        # Show best action arrow
                        best_a = np.argmax(qvals)
                        dx, dy = 0, 0
                        if best_a == 0: dy = -0.3 # N
                        elif best_a == 1: dx = 0.3 # S
                        elif best_a == 2: dy = 0.3 # E
                        elif best_a == 3: dx = -0.3 # W
                        ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.15, fc='blue', ec='blue')
            ax.set_title('Q-values and Policy Directions')
        Q = np.copy(self.Q)
        draw_grid(Q)
        plt.pause(0.5)
        for ep in range(episodes):
            state = self.env.start
            while not self.env.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.env.step(state, action)
                i, j = state
                ni, nj = next_state
                a_idx = self.env.actions.index(action)
                reward = -0.04 + self.env.rewards[next_state]
                best_next = np.max(self.Q[ni, nj])
                self.Q[i, j, a_idx] += self.alpha * (reward + self.gamma * best_next - self.Q[i, j, a_idx])
                state = next_state
            self.epsilon = max(self.epsilon * epsilon_decay, min_epsilon)
            if (ep + 1) % interval == 0 or ep == 0:
                Q = np.copy(self.Q)
                draw_grid(Q)
                plt.pause(0.1)
        Q = np.copy(self.Q)
        draw_grid(Q)
        plt.show()
    

    def get_policy(self):
        policy = np.full((self.env.height, self.env.width), '', dtype=object)
        for i in range(self.env.height):
            for j in range(self.env.width):
                if (i, j) in self.env.block:
                    policy[i, j] = 'X'
                elif (i, j) in self.env.terminal_states:
                    policy[i, j] = 'T'
                else:
                    policy[i, j] = self.env.actions[np.argmax(self.Q[i, j])]
        return policy
    
    def plot_learning_metrics(self, save_prefix=None):
        """Plot reward per episode and epsilon vs episodes on merged plot with dual y-axis"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot reward per episode on primary y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward per Episode', color=color1)
        line1 = ax1.plot(self.episode_numbers, self.episode_rewards, color=color1, alpha=0.7, linewidth=0.8, label='Episode Rewards')
        
        # Add moving average for smoother visualization
        window = min(100, len(self.episode_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            line2 = ax1.plot(self.episode_numbers[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving avg (window={window})')
        
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Create secondary y-axis for epsilon
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Epsilon (Exploration Rate)', color=color2)
        line3 = ax2.plot(self.episode_numbers, self.epsilon_history, color=color2, linewidth=2, label='Epsilon Decay')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add vertical dotted line at halfway point
        if self.episode_numbers:
            halfway_episode = max(self.episode_numbers) // 2
            ax1.axvline(x=halfway_episode, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Halfway ({halfway_episode} episodes)')
        
        # Combine legends from both axes
        lines = line1
        if window > 1:
            lines = line1 + line2
        lines = lines + line3
        labels = [l.get_label() for l in lines]
        if self.episode_numbers:
            labels.append(f'Halfway ({halfway_episode} episodes)')
        ax1.legend(lines + [plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.7)], labels, loc='best')
        
        plt.title(f'Learning Progress: Rewards & Epsilon Decay (α={self.alpha}, γ={self.gamma})')
        
        if save_prefix:
            plt.savefig(f'{save_prefix}_alpha_{self.alpha}_gamma_{self.gamma}_merged.png', dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        return fig

def run_hyperparameter_experiments(trap_penalty=-1, episodes=3000):
    """Run experiments with different alpha and gamma values - separate plot for each gamma"""
    alpha_values = [0.05, 0.1, 0.5, 0.7, 1.0]
    gamma_values = [0.2, 0.4, 0.6, 0.8, 0.95]
    
    # Define colors for different alpha values
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    results = {}
    
    # Create separate plot for each gamma value
    for gamma in gamma_values:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        print(f"\n=== Running experiments for γ={gamma} ===")
        
        # Run experiment for each alpha with current gamma
        for i, alpha in enumerate(alpha_values):
            print(f"Running: α={alpha}, γ={gamma}")
            
            env = GridWorld(trap_penalty=trap_penalty)
            agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0)
            
            # Learn with tracking
            agent.learn_with_tracking(episodes=episodes, epsilon_decay_rate=0.999, min_epsilon=0.01)
            
            # Store results
            key = f"alpha_{alpha}_gamma_{gamma}"
            results[key] = {
                'agent': agent,
                'policy': agent.get_policy(),
                'final_reward': agent.episode_rewards[-1] if agent.episode_rewards else 0,
                'avg_last_100': np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            }
            
            # Plot reward curve for this alpha
            # ax.plot(agent.episode_numbers, agent.episode_rewards, 
            #        color=colors[i], alpha=0.4, linewidth=0.8, 
            #        label=f'α={alpha} (raw)')
            
            # Add moving average
            window = min(100, len(agent.episode_rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
                ax.plot(agent.episode_numbers[window-1:], moving_avg, 
                       color=colors[i], linewidth=1, 
                       label=f'α={alpha} (avg)')
        
        # Add vertical line at halfway point
        if episodes > 0:
            halfway = episodes // 2
            ax.axvline(x=halfway, color='black', linestyle='--', alpha=0.7, 
                      linewidth=1.5, label=f'Halfway ({halfway})')
        
        # Customize plot
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward per Episode', fontsize=12)
        ax.set_title(f'Learning Curves for γ={gamma} (Different α values)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
        
        # Save individual plot
        plt.tight_layout()
        plt.savefig(f'plots/gamma_{gamma}_alpha_comparison_trap_{abs(trap_penalty)}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    
    for gamma in gamma_values:
        print(f"\nγ={gamma}:")
        gamma_results = {k: v for k, v in results.items() if f'gamma_{gamma}' in k}
        sorted_results = sorted(gamma_results.items(), key=lambda x: x[1]['avg_last_100'], reverse=True)
        
        for i, (key, result) in enumerate(sorted_results):
            alpha = key.split('_')[1]
            print(f"  {i+1}. α={alpha}: Avg last 100 = {result['avg_last_100']:.3f}")
    
    return results

def run_gamma_comparison_with_epsilon(alpha=0.1, trap_penalty=-1, episodes=3000):
    """Compare different gamma values with epsilon tracking - single alpha, multiple gammas"""
    gamma_values = [0.2, 0.4, 0.6, 0.8, 0.95]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    print(f"=== Running gamma comparison for α={alpha} ===")
    
    epsilon_data = None  # Will store epsilon from one run (they're all the same)
    
    # Run experiment for each gamma
    for i, gamma in enumerate(gamma_values):
        print(f"Running: α={alpha}, γ={gamma}")
        
        env = GridWorld(trap_penalty=trap_penalty)
        agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0)
        
        # Learn with tracking
        agent.learn_with_tracking(episodes=episodes, epsilon_decay_rate=0.999, min_epsilon=0.01)
        
        # Store epsilon data from first run (all runs have same epsilon pattern)
        if epsilon_data is None:
            epsilon_data = {
                'episodes': agent.episode_numbers.copy(),
                'epsilon': agent.epsilon_history.copy()
            }
        
        # Plot reward curve with moving average
        # Raw rewards (lighter)
        ax1.plot(agent.episode_numbers, agent.episode_rewards, 
                color=colors[i], alpha=0.1, linewidth=0.8)
        
        # Moving average (bold)
        window = min(100, len(agent.episode_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(agent.episode_numbers[window-1:], moving_avg, 
                    color=colors[i], linewidth=1, 
                    label=f'γ={gamma} (avg)')
    
    # Setup primary y-axis (rewards)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward per Episode', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for epsilon
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon (Exploration Rate)', color='darkred', fontsize=12)
    ax2.plot(epsilon_data['episodes'], epsilon_data['epsilon'], 
             color='darkred', linewidth=2, alpha=0.8, 
             label='Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Add vertical line at halfway point
    if episodes > 0:
        halfway = episodes // 2
        ax1.axvline(x=halfway, color='black', linestyle=':', alpha=0.7, 
                   linewidth=2, label=f'Halfway ({halfway})')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # Set title
    plt.title(f'Gamma Comparison: Rewards & Epsilon (α={alpha})', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'plots/gamma_comparison_alpha_{alpha}_with_epsilon_trap_{abs(trap_penalty)}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Quick single experiment
    env = GridWorld(trap_penalty=-200)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=1.0)

    agent.visualize_learning_matplotlib(epsilon_decay=0.999, min_epsilon=0.01, episodes=6000, interval=500)

    # print("Running single experiment with tracking...")
    # agent.learn_with_tracking(episodes=6000, epsilon_decay_rate=0.999, min_epsilon=0.01)
    
    # print("Policy after learning:")
    # policy = agent.get_policy()
    # print(policy)
    
    # # Plot metrics
    # agent.plot_learning_metrics(save_prefix="single_experiment")
    
    # # Uncomment to run full hyperparameter experiments
    # print("\nRunning hyperparameter experiments...")
    # # results = run_hyperparameter_experiments(trap_penalty=-1, episodes=6000)
    
    # # Run gamma comparison with epsilon tracking
    # print("\nRunning gamma comparison with epsilon tracking...")
    # run_gamma_comparison_with_epsilon(alpha=0.1, trap_penalty=-1, episodes=6000)