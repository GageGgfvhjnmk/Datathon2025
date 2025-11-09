"""
smart_train_with_early_stopping.py

Training script that automatically stops when reaching target win rate (70-75%)
"""

import time
import argparse
import random
from case_closed_game import Game, Direction
from local_judge import LocalJudge
from dqn_agent import SimpleDQNAgent
from emergency_dqn_agent import EmergencyPathfindingAgent
from tronbot_algorithm import get_tronbot_opponents

# Import our wrapper
try:
    from local_ram_wrapper import send_move_agent1, send_move_agent2
except ImportError:
    # Fallback if wrapper doesn't exist
    import local_ram
    def send_move_agent1(game):
        return local_ram.send_move_agent1(game)
    def send_move_agent2(game):
        # Simple fallback for agent2
        from case_closed_game import Direction
        return Direction.RIGHT, False


class SmartTrainer:
    def __init__(self, target_win_rate=60.0, patience=100):
        self.tronbot_opponents = get_tronbot_opponents()
        self.basic_opponents = [
            ("LocalRAM", self.local_ram_opponent),
            ("Aggressive", self.aggressive_opponent),
            ("Defensive", self.defensive_opponent),
            ("Random", self.random_opponent),
        ]
        self.all_opponents = self.basic_opponents + self.tronbot_opponents
        
        # Early stopping parameters
        self.target_win_rate = target_win_rate
        self.patience = patience  # Episodes to wait after reaching target
    
    def local_ram_opponent(self, game_state, my_agent, opponent):
        """Use local_ram logic for agent2"""
        try:
            return send_move_agent2(game_state)
        except Exception as e:
            print(f"⚠️  LocalRAM error: {e}, using fallback")
            return Direction.RIGHT, False
    
    def aggressive_opponent(self, game_state, my_agent, opponent):
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        current_direction = my_agent.direction
        reverse_directions = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        forbidden_direction = reverse_directions.get(current_direction)
        
        # Calculate direction to opponent
        dx = (opp_pos[0] - my_pos[0]) % game_state.board.width
        if dx > game_state.board.width // 2: 
            dx = dx - game_state.board.width
        dy = (opp_pos[1] - my_pos[1]) % game_state.board.height
        if dy > game_state.board.height // 2: 
            dy = dy - game_state.board.height
        
        # Prefer moves that get closer to opponent
        preferred_directions = []
        if abs(dx) > abs(dy):
            if dx > 0: 
                preferred_directions.append(Direction.RIGHT)
            else: 
                preferred_directions.append(Direction.LEFT)
            if dy > 0: 
                preferred_directions.append(Direction.DOWN)
            else: 
                preferred_directions.append(Direction.UP)
        else:
            if dy > 0: 
                preferred_directions.append(Direction.DOWN)
            else: 
                preferred_directions.append(Direction.UP)
            if dx > 0: 
                preferred_directions.append(Direction.RIGHT)
            else: 
                preferred_directions.append(Direction.LEFT)
        
        # Try preferred directions first
        for direction in preferred_directions:
            if direction != forbidden_direction:
                dx, dy = direction.value
                next_pos = (
                    (my_pos[0] + dx) % game_state.board.width,
                    (my_pos[1] + dy) % game_state.board.height
                )
                if (game_state.board.get_cell_state(next_pos) == 0 or 
                    next_pos == opp_pos):
                    # Use boost when close to opponent
                    dist = self._torus_manhattan(my_pos, opp_pos, game_state.board)
                    use_boost = (my_agent.boosts_remaining > 0 and dist < 4)
                    return direction, use_boost
        
        # Fallback to any safe move
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        for direction in directions:
            if direction != forbidden_direction:
                dx, dy = direction.value
                next_pos = (
                    (my_pos[0] + dx) % game_state.board.width,
                    (my_pos[1] + dy) % game_state.board.height
                )
                if (game_state.board.get_cell_state(next_pos) == 0 or 
                    next_pos == opp_pos):
                    return direction, False
        
        return current_direction, False
    
    def defensive_opponent(self, game_state, my_agent, opponent):
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        current_direction = my_agent.direction
        reverse_directions = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        forbidden_direction = reverse_directions.get(current_direction)
        
        # Calculate direction away from opponent
        dx = (opp_pos[0] - my_pos[0]) % game_state.board.width
        if dx > game_state.board.width // 2: 
            dx = dx - game_state.board.width
        dy = (opp_pos[1] - my_pos[1]) % game_state.board.height
        if dy > game_state.board.height // 2: 
            dy = dy - game_state.board.height
        
        # Prefer moves that increase distance from opponent
        preferred_directions = []
        if abs(dx) > abs(dy):
            if dx > 0: 
                preferred_directions.append(Direction.LEFT)
            else: 
                preferred_directions.append(Direction.RIGHT)
            if dy > 0: 
                preferred_directions.append(Direction.UP)
            else: 
                preferred_directions.append(Direction.DOWN)
        else:
            if dy > 0: 
                preferred_directions.append(Direction.UP)
            else: 
                preferred_directions.append(Direction.DOWN)
            if dx > 0: 
                preferred_directions.append(Direction.LEFT)
            else: 
                preferred_directions.append(Direction.RIGHT)
        
        # Try preferred directions first
        for direction in preferred_directions:
            if direction != forbidden_direction:
                dx, dy = direction.value
                next_pos = (
                    (my_pos[0] + dx) % game_state.board.width,
                    (my_pos[1] + dy) % game_state.board.height
                )
                if (game_state.board.get_cell_state(next_pos) == 0 or 
                    next_pos == opp_pos):
                    # Use boost to escape when opponent is close
                    dist = self._torus_manhattan(my_pos, opp_pos, game_state.board)
                    use_boost = (my_agent.boosts_remaining > 0 and dist < 3)
                    return direction, use_boost
        
        # Fallback to any safe move
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        for direction in directions:
            if direction != forbidden_direction:
                dx, dy = direction.value
                next_pos = (
                    (my_pos[0] + dx) % game_state.board.width,
                    (my_pos[1] + dy) % game_state.board.height
                )
                if (game_state.board.get_cell_state(next_pos) == 0 or 
                    next_pos == opp_pos):
                    return direction, False
        
        return current_direction, False
    
    def random_opponent(self, game_state, my_agent, opponent):
        current_direction = my_agent.direction
        reverse_directions = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        forbidden_direction = reverse_directions.get(current_direction)
        
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        valid_directions = []
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for direction in directions:
            if direction == forbidden_direction:
                continue
                
            dx, dy = direction.value
            next_pos = (
                (my_pos[0] + dx) % game_state.board.width,
                (my_pos[1] + dy) % game_state.board.height
            )
            if (game_state.board.get_cell_state(next_pos) == 0 or 
                next_pos == opp_pos):
                valid_directions.append(direction)
        
        if valid_directions:
            direction = random.choice(valid_directions)
            # Randomly use boost 20% of the time
            use_boost = (my_agent.boosts_remaining > 0 and random.random() < 0.2)
            return direction, use_boost
        else:
            return current_direction, False
    
    def _torus_manhattan(self, pos1, pos2, board):
        """Calculate Manhattan distance on torus board"""
        dx = min(abs(pos1[0] - pos2[0]), board.width - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), board.height - abs(pos1[1] - pos2[1]))
        return dx + dy
    
    def get_opponent(self, episode, total_episodes):
        progress = episode / total_episodes
        
        if progress < 0.2:
            opponent = random.choice(self.basic_opponents)
            return opponent, "🟢 BASIC"
        elif progress < 0.5:
            pool = self.basic_opponents + self.tronbot_opponents[:2]
            opponent = random.choice(pool)
            return opponent, "🟡 NOVICE"
        elif progress < 0.8:
            opponent = random.choice(self.tronbot_opponents)
            return opponent, "🟠 INTERMEDIATE"
        else:
            weights = [1, 1, 2, 2, 3]  # FloodFill, WallHugger, Minimax, SpaceInvader, Hybrid
            opponent = random.choices(self.tronbot_opponents, weights=weights)[0]
            return opponent, "🔴 EXPERT"


def smart_train_with_early_stopping(max_episodes=5000, target_win_rate=60.0, save_interval=100, 
                                   delay=0.05, max_turns=200, visuals=False, emergency_mode=True):
    """
    Smart training that stops automatically when reaching target win rate
    """
    print(f"🎯 Starting SMART training - Target: {target_win_rate}% win rate")
    print(f"🛑 Will stop automatically when target reached + 100 episodes patience")
    
    # Choose agent type
    if emergency_mode:
        agent = EmergencyPathfindingAgent(state_size=15, action_size=8)
        print("🆘 EMERGENCY PATHFINDING MODE: Enabled")
    else:
        agent = SimpleDQNAgent(state_size=15, action_size=8)
        print("🧠 STANDARD DQN MODE: Enabled")
    
    print(f"💻 Using device: {agent.device}")

    # Initialize counters
    wins = losses = draws = emergency_activations = 0
    judge = LocalJudge()
    trainer = SmartTrainer(target_win_rate=target_win_rate)
    
    # Early stopping tracking
    best_win_rate = 0
    episodes_above_target = 0
    target_reached = False
    stopped_early = False
    
    # Track opponent performance
    opponent_stats = {}
    for name, _ in trainer.all_opponents:
        opponent_stats[name] = [0, 0, 0]  # [wins, losses, draws]

    for ep in range(1, max_episodes + 1):
        # Reset game
        judge.game = Game()
        
        # Reset agent if needed
        try:
            agent.new_episode()
        except AttributeError:
            pass  # Some agents don't have new_episode method
        
        episode_reward = 0
        turns = 0
        episode_emergencies = 0

        # Select opponent
        (opponent_name, opponent_func), difficulty_level = trainer.get_opponent(ep, max_episodes)

        if visuals:
            print(f"\n🎮 Episode {ep}/{max_episodes} | Opponent: {opponent_name} | Level: {difficulty_level}")
            print(f"   Epsilon: {agent.epsilon:.3f}")

        # Game loop
        while judge.game.agent1.alive and judge.game.agent2.alive and judge.game.turns < max_turns:
            turns += 1
            
            # Visualize if enabled
            if visuals:
                judge.visualize_board()
                print(f"   Turn {turns} | Boosts: {judge.game.agent1.boosts_remaining}")
                if delay > 0:
                    time.sleep(delay)

            # RL AGENT (Player 1) - Get state and choose action
            state = agent.get_state_representation(judge.game, judge.game.agent1, judge.game.agent2)
            valid_actions = agent.get_valid_actions(
                judge.game, judge.game.agent1, judge.game.agent1.boosts_remaining, judge.game.agent2
            )
            
            # Choose action
            if not valid_actions:
                action = 0  # Fallback
                if visuals:
                    print("   ⚠️  No valid actions, using fallback")
            else:
                if emergency_mode and hasattr(agent, 'smart_act'):
                    # Use emergency pathfinding agent
                    action = agent.smart_act(state, valid_actions, judge.game, judge.game.agent1, judge.game.agent2)
                    # Check if emergency was triggered
                    if hasattr(agent, 'last_emergency_action') and agent.last_emergency_action == action:
                        episode_emergencies += 1
                        emergency_activations += 1
                        if visuals:
                            print("   🚨 EMERGENCY PATHFINDING ACTIVATED!")
                else:
                    # Use standard DQN agent
                    action = agent.act(state, valid_actions)

            # Validate action
            if valid_actions and action not in valid_actions:
                if visuals:
                    print(f"   ⚠️  Invalid action {action}, using random valid action")
                action = random.choice(valid_actions)

            # Convert action to direction and boost
            direction_idx = action // 2
            use_boost = (action % 2) == 1 and (judge.game.agent1.boosts_remaining > 0)
            direction = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP][direction_idx]
            action_used = (direction_idx, 1 if use_boost else 0)

            # OPPONENT (Player 2) - Get move from selected strategy
            try:
                opp_direction, opp_boost = opponent_func(judge.game, judge.game.agent2, judge.game.agent1)
            except Exception as e:
                if visuals:
                    print(f"   ❌ Opponent {opponent_name} error: {e}, using fallback")
                opp_direction, opp_boost = Direction.RIGHT, False

            # Store previous state for reward calculation
            prev_a1_alive = judge.game.agent1.alive
            prev_a2_alive = judge.game.agent2.alive

            # Execute both moves
            result = judge.game.step(direction, opp_direction, use_boost, opp_boost)

            # Calculate reward and get next state
            next_state = agent.get_state_representation(judge.game, judge.game.agent1, judge.game.agent2)
            reward = agent.get_reward(
                judge.game.agent1, judge.game.agent2, prev_a1_alive, prev_a2_alive, judge.game, action_used
            )
            done = not (judge.game.agent1.alive and judge.game.agent2.alive)

            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            episode_reward += reward

            # Check if game ended
            if result is not None:
                break

        # Episode results
        if not judge.game.agent1.alive and not judge.game.agent2.alive:
            draws += 1
            opponent_stats[opponent_name][2] += 1
            res = 'DRAW'
            emoji = '⚖️'
        elif not judge.game.agent1.alive:
            losses += 1
            opponent_stats[opponent_name][1] += 1
            res = 'LOSS'
            emoji = '💀'
        else:
            wins += 1
            opponent_stats[opponent_name][0] += 1
            res = 'WIN'
            emoji = '🏆'

        # Calculate current win rate
        current_win_rate = wins / ep * 100
        
        # 🎯 EARLY STOPPING LOGIC
        if current_win_rate >= target_win_rate:
            if not target_reached:
                print(f"🎯 TARGET REACHED! {current_win_rate:.1f}% win rate at episode {ep}")
                target_reached = True
                # Save the model when we first hit target
                agent.save_model()  # This will use the default path
                print(f"💾 Target model saved at episode {ep}")
            
            episodes_above_target += 1
            
            # Stop after maintaining target for 100 episodes
            # Replace the early stopping condition:
            if episodes_above_target >= 100 and ep >= 700:  # Only stop after episode 700
                print(f"🛑 EARLY STOPPING: Maintained {current_win_rate:.1f}%+ for 100 episodes (min 700 episodes)")
                stopped_early = True
                break
        else:
            # Reset counter if we drop below target
            episodes_above_target = 0
            target_reached = False

        # Update best win rate
        if current_win_rate > best_win_rate:
            best_win_rate = current_win_rate
            # Save best model
            agent.save_model("best_model_so_far.pth")
            if current_win_rate > target_win_rate - 10:  # Only print if close to target
                print(f"💾 New best model: {best_win_rate:.1f}%")

        # Episode statistics
        emergency_info = f" | Emergencies: {episode_emergencies}" if emergency_mode else ""
        
        # Only print every 50 episodes to reduce clutter, but always print milestones
        if ep % 50 == 0 or ep <= 10 or current_win_rate >= target_win_rate - 5:
            print(f"{emoji} Episode {ep}: {res} | Win Rate: {current_win_rate:.1f}% | Best: {best_win_rate:.1f}%{emergency_info}")

        # Save model periodically
        if ep % save_interval == 0:
            agent.save_model()
            print(f"💾 Checkpoint saved at episode {ep}")

    # Final save and report
    if not stopped_early:
        agent.save_model()
        print(f"💾 Final model saved (reached max episodes)")
    
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"📊 Final Stats: {wins} Wins, {losses} Losses, {draws} Draws")
    print(f"📈 Final Win Rate: {wins/ep*100:.1f}%")
    print(f"🏆 Best Win Rate: {best_win_rate:.1f}%")
    
    if stopped_early:
        print(f"✅ EARLY STOPPING: Successfully reached and maintained target performance!")
    else:
        print(f"⚠️  Max episodes reached - target not achieved")
    
    if emergency_mode:
        print(f"🆘 Total Emergency Activations: {emergency_activations}")
    
    # Opponent performance summary
    print(f"\n👥 Opponent Performance Summary:")
    for opponent_name, _ in trainer.all_opponents:
        stats = opponent_stats.get(opponent_name, [0, 0, 0])
        total_games = sum(stats)
        if total_games > 0:
            win_rate = stats[0] / total_games * 100
            print(f"   {opponent_name:12} {win_rate:5.1f}% win rate ({stats[0]:2}-{stats[1]:2}-{stats[2]:2})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart DQN Training with Early Stopping')
    parser.add_argument('--max-episodes', type=int, default=2000, help='Maximum episodes to train')
    parser.add_argument('--target-win-rate', type=float, default=60.0, help='Target win rate to stop at')
    parser.add_argument('--save-interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--delay', type=float, default=0.05, help='Visualization delay between moves')
    parser.add_argument('--no-visuals', dest='visuals', action='store_false', help='Disable visualization')
    parser.add_argument('--no-emergency', dest='emergency_mode', action='store_false', help='Disable emergency pathfinding')
    parser.set_defaults(visuals=False, emergency_mode=True)
    
    args = parser.parse_args()

    smart_train_with_early_stopping(
        max_episodes=args.max_episodes,
        target_win_rate=args.target_win_rate,
        save_interval=args.save_interval,
        delay=args.delay,
        visuals=args.visuals,
        emergency_mode=args.emergency_mode
    )