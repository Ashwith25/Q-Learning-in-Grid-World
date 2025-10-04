import os, sys
import random
from enum import Enum
from time import sleep
import copy

# import 

STOCHASTIC = (0.8, 0.2, 0.2)
# DETERMINISTIC = (1, 0, 0)
START_STATE = (2, 0)
GOAL_STATE = (0, 3)
GOAL_REWARD = 1
LOSE_STATE = (1, 3)
LOSE_REWARD = -1
WALLS = [(1, 1)]
ROWS = 3
COLS = 4
REWARDS = -0.04
    


class GridWorld:

    class action(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    def __init__(self, cols, rows, start, goal, goal_reward, lose, lose_reward, walls, rewards, probability=(1, 0, 0)):
        self.cols = cols
        self.rows = rows
        self.start = start
        self.goal = goal
        self.goal_reward = goal_reward
        self.lose_reward = lose_reward
        self.lose = lose
        self.walls = walls
        self.probability = probability
        self.grid = [['.' for _ in range(cols)] for _ in range(rows)]
        self.grid[goal[0]][goal[1]] = 'G'
        self.grid[lose[0]][lose[1]] = 'L'
        self.grid[start[0]][start[1]] = 'S'
        for wall in walls:
            self.grid[wall[0]][wall[1]] = '#'
        self.rewards = rewards
        self.agent_position = start
        self.grid[self.agent_position[0]][self.agent_position[1]] = 'A'  # Mark agent's position
        

    def is_terminal(self, state):
        return state == self.goal or state == self.lose

    def update_agent_position(self, new_position):
        # Update the grid to reflect the agent's new position
        if self.agent_position == self.start:
            self.grid[self.agent_position[0]][self.agent_position[1]] = 'S'  # Keep start marker
        elif self.agent_position == self.goal:
            self.grid[self.agent_position[0]][self.agent_position[1]] = 'G'  # Keep goal marker
        elif self.agent_position == self.lose:
            self.grid[self.agent_position[0]][self.agent_position[1]] = 'L'  # Keep lose marker
        else:
            self.grid[self.agent_position[0]][self.agent_position[1]] = '.'  # Clear old position
        self.agent_position = new_position
        self.grid[new_position[0]][new_position[1]] = 'A'  # Mark new position

    def move(self, state, action):
        if self.is_terminal(state):
            return state, -1  # No movement from terminal states

        row, col = state
        if action == self.action.UP:
            new_row, new_col = max(0, row - 1), col
        elif action == self.action.DOWN:
            new_row, new_col = min(self.rows - 1, row + 1), col
        elif action == self.action.LEFT:
            new_row, new_col = row, max(0, col - 1)
        elif action == self.action.RIGHT:
            new_row, new_col = row, min(self.cols - 1, col + 1)
        else:
            raise ValueError("Invalid action")

        if (new_row, new_col) in self.walls:
            self.update_agent_position(state)  # Stay in place if hitting a wall
            return state, self.rewards  # Hit a wall, stay in place

        new_state = (new_row, new_col)
        self.update_agent_position(new_state)
        if new_state == self.goal:
            return new_state, self.goal_reward  # Reward for reaching goal
        elif new_state == self.lose:
            return new_state, self.lose_reward  #lose state 
        else:
            return new_state, self.rewards 
        

    def display(self):
        #print the grid w the agent position
        print("Grid World:")
        #print indexes
        print('     ' + '     '.join(str(i) for i in range(self.cols)))
        for row in self.grid:
            print('   ', end ='')  # Align with column indexes
            print('----- ' * (self.cols))
            print(str(self.grid.index(row)) + ' ', end=' ')  # Print row index
            for cell in row:    
                print('|', end=' ')
                print(cell, end=' ')
                print('|', end=' ')
            print()


    # stocastic action selection
    def choose_action(self, action):
        #0 - up, 1 - down, 2 - left, 3 - right
        if action == action.UP:
            return random.choices([action.UP, action.LEFT, action.RIGHT], self.probability, k=1)[0] #returns integer
        elif action == action.DOWN:
            return random.choices([action.DOWN, action.LEFT, action.RIGHT], self.probability, k=1)[0]
        elif action == action.LEFT:
            return random.choices([action.LEFT, action.UP, action.DOWN], self.probability, k=1)[0]
        elif action == action.RIGHT:
            return random.choices([action.RIGHT, action.UP, action.DOWN], self.probability, k=1)[0]
        else:
            raise ValueError("Invalid action")



class QLearningAgent:
    def __init__(self, env, actions, alpha=0.7, gamma=0.9, epsilon=1.0):
        self.env = env
        self.q_table = {}  # state-action value table
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99
        self.actions = actions

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)  # Default Q-value is 0.0

    def choose_action(self, state):
        # Explore
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values, default=0.0)

        # collect all actions that yield max_q
        max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]

        return random.choice(max_actions)

    def update(self, state, action, reward, next_state):
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions], default=0.0) #Q argmax a
        current_q = self.get_q_value(state, action)
        print(f"Current Q: {current_q}, Reward: {reward}, Max Next Q: {max_next_q}")
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)  # min epsilon decay = 0.01, max decay = 0.99
        if self.epsilon == self.min_epsilon:
            print("Epsilon has decayed to its minimum value.")
            # sleep(10)

    def display_q_table(self):
        print("Q-Table:")
        # Print states in row-major order and actions in a fixed, readable order.
        # We'll skip wall positions and print a default Q-value of 0.0 when missing.
        action_order = [self.env.action.UP, self.env.action.DOWN, self.env.action.RIGHT, self.env.action.LEFT]
        action_short = {'UP': 'N', 'DOWN': 'S', 'LEFT': 'W', 'RIGHT': 'E'}
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)
                # skip walls (optional) so output focuses on valid states
                if state in getattr(self.env, 'walls', []):
                    continue
                values = []
                for a in action_order:
                    q = self.get_q_value(state, a)
                    values.append(f"{action_short[a.name]}: {q:.3f}")
                print(f"{state} - " + ", ".join(values))
        

    def display_optimal_policy(self):
        print("Optimal Policy:")
        action_short = {self.env.action.UP: 'N', self.env.action.DOWN: 'S', self.env.action.LEFT: 'W', self.env.action.RIGHT: 'E'}
        print('     ' + '     '.join(str(i) for i in range(self.env.cols)))
        r = 0
        for row in range(self.env.rows):
            print('   ', end ='')  # Align with column indexes
            print('----- ' * (self.env.cols))
            print(str(r) + ' ', end=' ')  # Print row index
            r += 1
            for col in range(self.env.cols):
                state = (row, col)
                # skip walls (optional) so output focuses on valid states
                if state in getattr(self.env, 'walls', []):
                    print('|', end=' ')
                    print("X", end=" ")
                    print('|', end=' ')
                    continue
                q_values = [self.get_q_value(state, a) for a in self.actions]
                max_q = max(q_values, default=0.0)
                max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
                print('|', end=' ')
                if len(max_actions) == 1:
                    short = action_short[max_actions[0]] if max_actions else " "
                    print(short, end=" ")
                else:
                    print("T", end=" ")  # Tie
                print('|', end=' ')
            print()

    def check_convergence(self, Q_prev, threshold=0.01):
        max_q_change = 0.0
        for state in [(r, c) for r in range(self.env.rows) for c in range(self.env.cols) if (r, c) not in self.env.walls]:
            for action in self.actions:
                current_q = self.get_q_value(state, action)
                prev_q = Q_prev.get((state, action), 0.0)
                change = abs(current_q - prev_q)
                if change > max_q_change:
                    max_q_change = change

        if max_q_change < threshold:
            print(f"Converged! Max Q-value change: {max_q_change:.5f}")
            return True

        return False

    def check_policy_convergence(self):
        # Check if the optimal policy has stabilized
        stable = True
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                state = (r, c)
                if state in getattr(self.env, 'walls', []):
                    continue
                q_values = [self.get_q_value(state, a) for a in self.actions]
                max_q = max(q_values, default=0.0)
                max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
                if len(max_actions) > 1:
                    stable = False  # There's a tie, so policy isn't stable
        if stable:
            print("Policy has stabilized.")
        return stable


# print(GOAL_STATE[0], GOAL_STATE[1])
gw = GridWorld(COLS, ROWS, START_STATE, GOAL_STATE, GOAL_REWARD, LOSE_STATE, LOSE_REWARD, WALLS, REWARDS, STOCHASTIC)
# print(action.UP)
# print(gw.choose_action(GridWorld.action.UP))
agent = QLearningAgent(gw, list(GridWorld.action))

def simulate_episode(env, agent, max_steps=100):
    state = env.start
    total_reward = 0
    steps = 0
    print("Starting new episode. Grid World:")
    env.display()
    print("Initial Q-Table:")
    agent.display_q_table()
    while not env.is_terminal(state) and steps < max_steps:
        action = agent.choose_action(state)
        chosen_action = env.choose_action(action)  # Stochastic action selection
        next_state, reward = env.move(state, chosen_action)
        agent.update(state, action, reward, next_state) #agent should update based on intended action, not stochastic action
        state = next_state
        total_reward += reward
        steps += 1
        # print(f"Step {steps}, State: {state}, Total Reward: {total_reward}")
        # env.display()
        # agent.display_q_table()

    return total_reward

# simulate_episode(gw, agent, max_steps=50)
def train_agent(env, agent, episodes=100):
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")
        q_prev = copy.deepcopy(agent.q_table)
        total_reward = simulate_episode(env, agent)
        agent.display_q_table()
        env.display()
        agent.display_optimal_policy()
        agent.decay_epsilon()
        if agent.check_convergence(q_prev, 0.01):
            print(f"Training converged after {episode + 1} episodes.")
            break
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        # sleep(2)
        # input("Press Enter to continue to the next episode...")



train_agent(gw, agent, episodes=20_000)
print("Final Q-Table:")
agent.display_q_table()
agent.display_optimal_policy()
print("Final Grid World:")
gw.display()