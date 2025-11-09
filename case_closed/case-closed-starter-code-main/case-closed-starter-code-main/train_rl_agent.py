"""
Training script for RL agent
Trains the heuristic-guided DQN through self-play against the heuristic
Now with visualization to watch the agent learn!
"""

import torch
from case_closed_game import Game, GameResult
import agent_strategies
import RL_agent_strategy
from collections import deque
import time
import os


class RLTrainer:
    """Trains RL agent through self-play"""
    
    def __init__(self, episodes=1000, save_freq=50, visualize_freq=0):
        self.episodes = episodes
        self.save_freq = save_freq
        self.visualize_freq = visualize_freq  # Visualize every N episodes (0 = never)
        self.rl_agent = RL_agent_strategy.get_rl_agent()
        self.rl_agent.training_mode = True
        
        # Training stats
        self.win_history = deque(maxlen=100)
        self.score_history = deque(maxlen=100)
    
    def visualize_board(self, game, turn, episode_num):
        """Display the current game state with colored output"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        board = [[' ' for _ in range(game.board.width)] for _ in range(game.board.height)]
        
        # Mark agent 1 trail (green - Heuristic)
        for i, pos in enumerate(game.agent1.trail):
            x, y = pos
            if i == len(game.agent1.trail) - 1:  # Head
                board[y][x] = '\033[92mH\033[0m'  # H for Heuristic
            else:
                board[y][x] = '\033[92m█\033[0m'
        
        # Mark agent 2 trail (cyan - RL)
        for i, pos in enumerate(game.agent2.trail):
            x, y = pos
            if i == len(game.agent2.trail) - 1:  # Head
                board[y][x] = '\033[96mR\033[0m'  # R for RL
            else:
                board[y][x] = '\033[96m█\033[0m'
        
        # Print board
        print("=" * (game.board.width * 2 + 2))
        print(f"TRAINING Episode {episode_num} | Turn: {turn}")
        print(f"Heuristic (H/Green): {game.agent1.length} trail, {game.agent1.boosts_remaining} boosts")
        print(f"RL Agent  (R/Cyan):  {game.agent2.length} trail, {game.agent2.boosts_remaining} boosts | ε={self.rl_agent.epsilon:.3f}")
        print("=" * (game.board.width * 2 + 2))
        for row in board:
            print('|' + ''.join(cell if cell != ' ' else '.' for cell in row) + '|')
        print("=" * (game.board.width * 2 + 2))
        
    def check_trapped_in_negative_space(self, game, player_number, min_space=15):
        """
        Check if a player is trapped in a negative space (enclosed area too small)
        Returns: (is_trapped, available_space)
        """
        if player_number == 1:
            my_trail = game.agent1.trail
            my_player_id = 1
        else:
            my_trail = game.agent2.trail
            my_player_id = 2
        
        if not my_trail:
            return False, 0
        
        # Current head position
        head_pos = tuple(my_trail[-1])
        my_trail_set = set(tuple(pos) for pos in my_trail)
        
        # Flood fill to count available space
        width = game.board.width
        height = game.board.height
        board = game.board.grid
        
        visited = set()
        queue = [head_pos]
        visited.add(head_pos)
        space_count = 0
        
        dir_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        while queue and space_count <= min_space + 5:  # Stop early if enough space
            cx, cy = queue.pop(0)
            space_count += 1
            
            for dx, dy in dir_map:
                nx = (cx + dx) % width
                ny = (cy + dy) % height
                
                if (nx, ny) not in visited:
                    # Check if cell is free (value 0 = empty)
                    if board[ny][nx] == 0 and (nx, ny) not in my_trail_set:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        is_trapped = space_count < min_space
        return is_trapped, space_count
    
    def run_training_episode(self, episode_num, visualize=False):
        """Run one training episode with detailed reward tracking"""
        game = Game()
        states = []
        actions = []
        rewards_partial = []
        
        result = None
        turn = 0
        max_turns = 200
        
        # Track positions for death reason detection
        agent1_next_pos = None
        agent2_next_pos = None
        agent1_will_hit_own = False
        agent1_will_hit_other = False
        agent2_will_hit_own = False
        agent2_will_hit_other = False
        
        # Track trap detection
        agent2_was_trapped = False
        agent2_trap_space = 0
        
        # Optional visualization delay
        delay = 0.05 if visualize else 0
        
        while result is None and turn < max_turns:
            turn += 1
            
            # Visualize current state
            if visualize:
                self.visualize_board(game, turn, episode_num)
                time.sleep(delay)
            
            # Get current state for RL agent (Agent 2)
            state_snapshot = {
                'board': [row[:] for row in game.board.grid],
                'agent1_trail': list(game.agent1.trail),
                'agent2_trail': list(game.agent2.trail),
                'turn': turn
            }
            
            # Agent 1: Heuristic
            dir1, boost1 = agent_strategies.send_move_agent1(game)
            
            # Agent 2: RL (with exploration)
            dir2, boost2 = self.rl_agent.select_action(game, player_number=2, training=True)
            
            # Store state and action for RL agent
            states.append(state_snapshot)
            actions.append((dir2, boost2))
            
            # Track next positions BEFORE executing move
            agent1_next_pos = tuple(game.agent1.trail[-1])
            agent2_next_pos = tuple(game.agent2.trail[-1])
            
            # Simulate moves to predict positions
            dx1, dy1 = dir1.value
            steps1 = 2 if boost1 else 1
            for _ in range(steps1):
                agent1_next_pos = ((agent1_next_pos[0] + dx1) % game.board.width,
                                   (agent1_next_pos[1] + dy1) % game.board.height)
            
            dx2, dy2 = dir2.value
            steps2 = 2 if boost2 else 1
            for _ in range(steps2):
                agent2_next_pos = ((agent2_next_pos[0] + dx2) % game.board.width,
                                   (agent2_next_pos[1] + dy2) % game.board.height)
            
            # Check collision reasons BEFORE executing
            agent1_will_hit_own = agent1_next_pos in game.agent1.trail
            agent1_will_hit_other = agent1_next_pos in game.agent2.trail
            agent2_will_hit_own = agent2_next_pos in game.agent2.trail
            agent2_will_hit_other = agent2_next_pos in game.agent1.trail
            
            # Execute move
            result = game.step(dir1, dir2, boost1, boost2)
            
            # Check if agent 2 (RL) got trapped after this move
            if result is None:  # Game still ongoing
                is_trapped, trap_space = self.check_trapped_in_negative_space(game, player_number=2, min_space=15)
                if is_trapped:
                    agent2_was_trapped = True
                    agent2_trap_space = trap_space
            
            # Intermediate reward (small penalty for survival to encourage winning)
            intermediate_reward = -0.1
            rewards_partial.append(intermediate_reward)
        
        # Show final state if visualizing
        if visualize:
            self.visualize_board(game, turn, episode_num)
            time.sleep(0.5)  # Pause to see final state
        
        # === DETAILED REWARD BASED ON DEATH REASON ===
        final_reward = 0
        death_reason = ""
        
        if result == GameResult.AGENT2_WIN:
            # RL agent (Agent 2) WINS!
            if agent1_will_hit_other:
                # Opponent crossed RL's path
                final_reward = 25
                death_reason = "Opponent crossed RL's path (+25)"
            elif agent1_will_hit_own:
                # Opponent crossed its own path
                final_reward = 10
                death_reason = "Opponent crossed itself (+10)"
            else:
                # Default win (e.g., invalid move)
                final_reward = 25
                death_reason = "Win (opponent invalid move, +25)"
            self.win_history.append(1)
            
        elif result == GameResult.AGENT1_WIN:
            # RL agent (Agent 2) LOSES
            if agent2_was_trapped:
                # RL TRAPPED ITSELF in negative space - severe penalty!
                final_reward = -100
                death_reason = f"RL TRAPPED itself (only {agent2_trap_space} space, -100)"
            elif agent2_will_hit_other:
                # RL crossed opponent's path
                final_reward = -10
                death_reason = "RL crossed opponent's path (-10)"
            elif agent2_will_hit_own:
                # RL crossed its own path
                final_reward = -25
                death_reason = "RL crossed itself (-25)"
            else:
                # Default loss (e.g., invalid move)
                final_reward = -25
                death_reason = "Loss (RL invalid move, -25)"
            self.win_history.append(0)
            
        else:
            # Draw
            final_reward = 0
            death_reason = "Draw (0)"
            self.win_history.append(0.5)
        
        # Calculate cumulative rewards (discounted from final reward)
        gamma = 0.95
        rewards = []
        cumulative = final_reward
        for i in range(len(rewards_partial) - 1, -1, -1):
            cumulative = rewards_partial[i] + gamma * cumulative
            rewards.insert(0, cumulative)
        
        total_score = final_reward + sum(rewards_partial)
        self.score_history.append(total_score)
        
        # Store experiences in replay memory
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = states[i + 1] if i + 1 < len(states) else None
            done = (i == len(states) - 1)
            
            # Simplified storage (full implementation would convert to tensors)
            # self.rl_agent.store_experience(state, action, reward, next_state, done)
        
        # Train on batch
        # loss = self.rl_agent.train_step()
        
        # Show result if visualizing
        if visualize:
            result_str = ""
            if result == GameResult.AGENT2_WIN:
                result_str = "\033[92m🎉 RL AGENT WINS!\033[0m"
            elif result == GameResult.AGENT1_WIN:
                result_str = "\033[91m❌ RL AGENT LOST\033[0m"
            else:
                result_str = "🤝 DRAW"
            
            print(f"\n{result_str}")
            print(f"Reason: {death_reason}")
            print(f"Final Score: {total_score:.1f}")
            time.sleep(2)  # Pause to see result
        
        return result, total_score, turn, death_reason
    
    def train(self):
        """Main training loop"""
        print(f"Starting RL Training for {self.episodes} episodes")
        print(f"Device: {self.rl_agent.device}")
        print(f"Initial epsilon: {self.rl_agent.epsilon:.3f}")
        if self.visualize_freq > 0:
            print(f"Visualization: Every {self.visualize_freq} episodes")
        print("=" * 60)
        
        start_time = time.time()
        
        # Track reward distribution
        reward_counts = {25: 0, 10: 0, 0: 0, -10: 0, -25: 0, -100: 0}
        
        for episode in range(1, self.episodes + 1):
            # Determine if we should visualize this episode
            should_visualize = (self.visualize_freq > 0 and episode % self.visualize_freq == 0)
            
            result, score, turns, death_reason = self.run_training_episode(episode, visualize=should_visualize)
            
            # Track final reward distribution
            if "-100" in death_reason:
                reward_counts[-100] += 1
            elif "+25" in death_reason:
                reward_counts[25] += 1
            elif "+10" in death_reason:
                reward_counts[10] += 1
            elif "-10" in death_reason:
                reward_counts[-10] += 1
            elif "-25" in death_reason:
                reward_counts[-25] += 1
            else:
                reward_counts[0] += 1
            
            # Print progress
            if episode % 10 == 0:
                win_rate = sum(self.win_history) / len(self.win_history) if self.win_history else 0
                avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
                
                result_str = "WIN" if result == GameResult.AGENT2_WIN else "LOSS" if result == GameResult.AGENT1_WIN else "DRAW"
                
                print(f"Episode {episode:4d} | {result_str:4s} | "
                      f"Turns: {turns:3d} | Score: {score:6.1f} | "
                      f"Epsilon: {self.rl_agent.epsilon:.3f} | "
                      f"Win Rate: {win_rate:.2%} | Avg Score: {avg_score:6.1f}")
                print(f"  → {death_reason}")
            
            # Save model periodically
            if episode % self.save_freq == 0:
                self.rl_agent.save_model(f"rl_agent_episode_{episode}.pth")
                elapsed = time.time() - start_time
                print(f"  → Model saved at episode {episode} (elapsed: {elapsed:.1f}s)")
                
                # Print reward distribution
                print(f"  → Reward distribution: +25={reward_counts[25]}, +10={reward_counts[10]}, "
                      f"0={reward_counts[0]}, -10={reward_counts[-10]}, -25={reward_counts[-25]}, -100={reward_counts[-100]}")
        
        # Final save
        self.rl_agent.save_model("rl_agent_final.pth")
        
        total_time = time.time() - start_time
        final_win_rate = sum(self.win_history) / len(self.win_history) if self.win_history else 0
        
        print("=" * 60)
        print(f"Training Complete!")
        print(f"Total time: {total_time:.1f}s ({total_time/self.episodes:.2f}s per episode)")
        print(f"Final win rate (last 100): {final_win_rate:.2%}")
        print(f"Final epsilon: {self.rl_agent.epsilon:.3f}")
        print()
        print(f"Final Reward Distribution:")
        print(f"  +25 (Opponent hit RL's trail):    {reward_counts[25]:4d} ({reward_counts[25]/self.episodes*100:5.1f}%)")
        print(f"  +10 (Opponent hit own trail):     {reward_counts[10]:4d} ({reward_counts[10]/self.episodes*100:5.1f}%)")
        print(f"    0 (Draw):                        {reward_counts[0]:4d} ({reward_counts[0]/self.episodes*100:5.1f}%)")
        print(f"  -10 (RL hit opponent's trail):    {reward_counts[-10]:4d} ({reward_counts[-10]/self.episodes*100:5.1f}%)")
        print(f"  -25 (RL hit own trail):           {reward_counts[-25]:4d} ({reward_counts[-25]/self.episodes*100:5.1f}%)")
        print(f" -100 (RL TRAPPED in neg. space):  {reward_counts[-100]:4d} ({reward_counts[-100]/self.episodes*100:5.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    # Training configuration
    print("=" * 60)
    print("RL AGENT TRAINING")
    print("=" * 60)
    print()
    print("Choose visualization mode:")
    print("  0 - No visualization (fastest training)")
    print("  1 - Visualize final episode only")
    print(" 10 - Visualize every 10th episode")
    print(" 50 - Visualize every 50th episode")
    print()
    
    try:
        viz_choice = input("Enter visualization frequency (default: 0): ").strip()
        visualize_freq = int(viz_choice) if viz_choice else 0
    except ValueError:
        visualize_freq = 0
    
    print()
    print(f"Starting training with visualization every {visualize_freq} episodes" if visualize_freq > 0 else "Starting training without visualization")
    print()
    
    trainer = RLTrainer(
        episodes=10000,              # Number of training games
        save_freq=1000,              # Save model every N episodes
        visualize_freq=1000  # Visualize every N episodes
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model...")
        trainer.rl_agent.save_model("rl_agent_interrupted.pth")
        print("Model saved!")
