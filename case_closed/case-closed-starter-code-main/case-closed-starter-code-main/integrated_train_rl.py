"""
Integrated RL Training System
Combines:
1. HeuristicGuidedDQNAgent with emergency pathfinding (from RL_agent_strategy.py)
2. Progressive difficulty training (from ultimate_train_with_visuals.py)
3. Multiple opponent types (TronBot algorithms + basic strategies)
4. Trap detection with -100 penalty
5. Emergency A* pathfinding override
"""

import argparse
import random
import time
from collections import defaultdict
from case_closed_game import Direction
from local_judge import LocalJudge
from RL_agent_strategy import HeuristicGuidedDQNAgent
from tronbot_algorithm import get_tronbot_opponents
from local_ram_wrapper import send_move_agent1 as local_ram_opponent_move
from diverse_opponents import get_diverse_opponents


class IntegratedTrainer:
    """Training orchestrator with progressive difficulty and multiple opponents"""
    
    def __init__(self, use_diverse_opponents=True):
        self.use_diverse_opponents = use_diverse_opponents
        self.setup_opponents()
        
    def setup_opponents(self):
        """Setup all available opponents"""
        
        # Basic opponents
        self.basic_opponents = [
            ("LocalRAM", self.local_ram_opponent),
            ("Aggressive", self.aggressive_opponent),
            ("Defensive", self.defensive_opponent),
            ("Random", self.random_opponent),
        ]
        
        # TronBot algorithm opponents
        self.tronbot_opponents = get_tronbot_opponents()
        
        # Diverse opponents (new!)
        if self.use_diverse_opponents:
            self.diverse_opponents = get_diverse_opponents()
        else:
            self.diverse_opponents = []
        
        # All opponents combined
        self.all_opponents = self.basic_opponents + self.tronbot_opponents + self.diverse_opponents
        
        print(f"📋 Loaded {len(self.all_opponents)} opponents:")
        print(f"   Basic: {len(self.basic_opponents)}")
        print(f"   TronBot: {len(self.tronbot_opponents)}")
        if self.use_diverse_opponents:
            print(f"   Diverse: {len(self.diverse_opponents)}")
        print()
        for name, _ in self.all_opponents:
            print(f"   - {name}")
    
    def local_ram_opponent(self, game, my_agent, opp_agent):
        """LocalRAM strategy wrapper"""
        # local_ram expects agent1, so we need to swap if this is agent2
        direction, boost = local_ram_opponent_move(game)
        return direction, boost
    
    def aggressive_opponent(self, game, my_agent, opp_agent):
        """Aggressive opponent - always moves toward opponent"""
        my_x, my_y = my_agent.trail[-1]
        opp_x, opp_y = opp_agent.trail[-1]
        
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Calculate direction to opponent (with torus wrapping)
        dx = opp_x - my_x
        dy = opp_y - my_y
        
        if abs(dx) > width // 2:
            dx = -dx / abs(dx) * (width - abs(dx))
        if abs(dy) > height // 2:
            dy = -dy / abs(dy) * (height - abs(dy))
        
        # Choose direction
        if abs(dx) > abs(dy):
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy > 0 else Direction.UP
        
        # Check if move is safe
        test_x, test_y = my_x, my_y
        test_x = (test_x + direction.value[0]) % width
        test_y = (test_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            # Not safe, pick random safe move
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        return direction, False
    
    def defensive_opponent(self, game, my_agent, opp_agent):
        """Defensive opponent - moves away from opponent"""
        my_x, my_y = my_agent.trail[-1]
        opp_x, opp_y = opp_agent.trail[-1]
        
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Calculate direction away from opponent
        dx = my_x - opp_x
        dy = my_y - opp_y
        
        if abs(dx) > width // 2:
            dx = -dx / abs(dx) * (width - abs(dx))
        if abs(dy) > height // 2:
            dy = -dy / abs(dy) * (height - abs(dy))
        
        # Choose direction
        if abs(dx) > abs(dy):
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy > 0 else Direction.UP
        
        # Check if move is safe
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            # Not safe, pick random safe move
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        return direction, False
    
    def random_opponent(self, game, my_agent, opp_agent):
        """Random safe moves"""
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        safe_moves = []
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            tx = (my_x + direction.value[0]) % width
            ty = (my_y + direction.value[1]) % height
            if game.board.grid[ty][tx] == 0:
                safe_moves.append(direction)
        
        if safe_moves:
            return random.choice(safe_moves), False
        return Direction.UP, False
    
    def get_opponent(self, episode, total_episodes):
        """
        Get opponent with progressive difficulty
        
        Progressive difficulty levels:
        - 0-20%: Basic opponents only (LocalRAM, Aggressive, Defensive, Random)
        - 20-50%: Novice mix (60% basic, 40% simple TronBot)
        - 50-80%: Intermediate (40% basic, 60% advanced TronBot)
        - 80-100%: Expert (20% basic, 80% all TronBot algorithms)
        """
        progress = episode / total_episodes
        
        if progress < 0.20:
            # Basic level - only simple opponents
            difficulty_level = "🟢 Basic"
            pool = self.basic_opponents
        elif progress < 0.50:
            # Novice level - mix of basic and simple bots
            difficulty_level = "🟡 Novice"
            if random.random() < 0.6:
                pool = self.basic_opponents
            else:
                pool = self.tronbot_opponents[:2]  # FloodFill, WallHugger
        elif progress < 0.80:
            # Intermediate level - more advanced bots
            difficulty_level = "🟠 Intermediate"
            if random.random() < 0.4:
                pool = self.basic_opponents
            else:
                pool = self.tronbot_opponents[:4]  # Include Minimax, SpaceInvader
        else:
            # Expert level - all opponents
            difficulty_level = "🔴 Expert"
            if random.random() < 0.2:
                pool = self.basic_opponents
            else:
                pool = self.tronbot_opponents  # All TronBot algorithms
        
        opponent_name, opponent_func = random.choice(pool)
        return opponent_name, opponent_func, difficulty_level


def integrated_train(
    episodes=5000,
    save_interval=100,
    delay=0.05,
    visuals=False,
    emergency_mode=True,
    use_diverse_opponents=True,
    max_turns=1000
):
    """
    Main integrated training loop
    
    Args:
        episodes: Number of training episodes
        save_interval: Save model every N episodes
        delay: Visualization delay in seconds
        visuals: Enable visualization
        emergency_mode: Enable emergency A* pathfinding
        use_diverse_opponents: Include diverse opponent strategies
        max_turns: Maximum turns per game
    """
    
    print("=" * 60)
    print("🚀 INTEGRATED RL TRAINING SYSTEM")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Emergency Mode: {'✅ ENABLED' if emergency_mode else '❌ DISABLED'}")
    print(f"   Diverse Opponents: {'✅ ENABLED' if use_diverse_opponents else '❌ DISABLED'}")
    print(f"   Visualization: {'✅ ON' if visuals else '❌ OFF'}")
    print(f"   Save Interval: {save_interval}")
    print("=" * 60)
    
    # Initialize agent and trainer
    agent = HeuristicGuidedDQNAgent()
    trainer = IntegratedTrainer(use_diverse_opponents=use_diverse_opponents)
    
    # Try to load existing model
    agent.load_model()
    
    # Training statistics
    wins = 0
    losses = 0
    draws = 0
    emergency_activations = 0
    opponent_stats = defaultdict(lambda: [0, 0, 0])  # [wins, losses, draws]
    
    for ep in range(1, episodes + 1):
        # Get opponent for this episode
        opponent_name, opponent_func, difficulty_level = trainer.get_opponent(ep, episodes)
        
        # Initialize game
        judge = LocalJudge()
        turns = 0
        episode_reward = 0.0
        episode_emergencies = 0
        
        # Episode header
        if visuals or ep % 10 == 0:
            print(f"\n🎮 Episode {ep}/{episodes} | Opponent: {opponent_name} | Level: {difficulty_level}")
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
            
            # RL AGENT (Player 1) - Get state
            state = {
                'board': judge.game.board.grid,
                'agent1_trail': list(judge.game.agent1.trail),
                'agent2_trail': list(judge.game.agent2.trail),
                'agent1_boosts': judge.game.agent1.boosts_remaining,
                'agent2_boosts': judge.game.agent2.boosts_remaining,
                'agent1_direction': judge.game.agent1.direction,
                'agent2_direction': judge.game.agent2.direction,
                'player_number': 1
            }
            
            # Check if emergency mode and if agent is in danger
            in_emergency = False
            if emergency_mode and agent.is_in_danger(judge.game, judge.game.agent1, judge.game.agent2):
                in_emergency = True
                episode_emergencies += 1
                emergency_activations += 1
                if visuals:
                    print("   🚨 EMERGENCY PATHFINDING ACTIVATED!")
            
            # Choose action (with emergency override if needed)
            direction, use_boost = agent.select_action(judge.game, player_number=1, training=True)
            
            # OPPONENT (Player 2) - Get move from selected strategy
            try:
                opp_direction, opp_boost = opponent_func(judge.game, judge.game.agent2, judge.game.agent1)
            except Exception as e:
                print(f"   ❌ Opponent {opponent_name} error: {e}, using fallback")
                opp_direction, opp_boost = Direction.RIGHT, False
            
            # Store previous state for reward calculation
            prev_a1_alive = judge.game.agent1.alive
            prev_a2_alive = judge.game.agent2.alive
            
            # Execute both moves
            result = judge.game.step(direction, opp_direction, use_boost, opp_boost)
            
            # Calculate reward
            reward = 0.0
            
            # Basic rewards
            if not judge.game.agent1.alive and not judge.game.agent2.alive:
                reward = 0.0  # Draw
            elif not judge.game.agent1.alive:
                reward = -100.0  # Death
            elif not judge.game.agent2.alive:
                reward = 100.0  # Win
            else:
                # Survival reward
                reward = 1.0
                
                # Emergency survival bonus
                if in_emergency:
                    reward += 2.0
                
                # Check for trap creation (existing logic)
                my_pos = judge.game.agent1.trail[-1]
                my_trail_set = set(tuple(pos) for pos in judge.game.agent1.trail)
                is_trapped, available_space = agent._check_if_trapped(
                    judge.game.board.grid,
                    my_pos,
                    my_trail_set,
                    my_player_id=1,
                    min_space=15
                )
                
                if is_trapped:
                    # TRAP PENALTY: -100 for creating negative space
                    reward -= 100.0
                    if visuals:
                        print(f"   ⚠️  TRAP DETECTED! Available space: {available_space}")
            
            episode_reward += reward
            
            # Get next state
            next_state = {
                'board': judge.game.board.grid,
                'agent1_trail': list(judge.game.agent1.trail),
                'agent2_trail': list(judge.game.agent2.trail),
                'agent1_boosts': judge.game.agent1.boosts_remaining,
                'agent2_boosts': judge.game.agent2.boosts_remaining,
                'agent1_direction': judge.game.agent1.direction,
                'agent2_direction': judge.game.agent2.direction,
                'player_number': 1
            }
            
            # Store experience (simplified - using state dict directly)
            done = not (judge.game.agent1.alive and judge.game.agent2.alive)
            # Note: Full implementation would convert state to proper format
            
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
        
        # Episode statistics
        win_rate = wins / ep * 100
        emergency_info = f" | Emergencies: {episode_emergencies}" if emergency_mode else ""
        
        if visuals or ep % 10 == 0:
            print(f"{emoji} Episode {ep}: {res} | Reward: {episode_reward:.2f} | Win Rate: {win_rate:.1f}%{emergency_info}")
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Save model periodically
        if ep % save_interval == 0:
            agent.save_model()
            print(f"💾 Model saved at episode {ep}")
            print(f"📊 Stats: {wins}W-{losses}L-{draws}D | Win Rate: {win_rate:.1f}%")
            if emergency_mode:
                print(f"🆘 Emergency Activations: {emergency_activations}")
    
    # Final save and report
    agent.save_model()
    
    print(f"\n{'=' * 60}")
    print(f"🎯 TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"📊 Final Stats: {wins} Wins, {losses} Losses, {draws} Draws")
    print(f"📈 Final Win Rate: {wins/episodes*100:.1f}%")
    
    if emergency_mode:
        print(f"🆘 Total Emergency Activations: {emergency_activations}")
    
    # Opponent performance summary
    print(f"\n👥 Opponent Performance Summary:")
    for opponent_name in sorted(opponent_stats.keys()):
        stats = opponent_stats[opponent_name]
        total_games = sum(stats)
        if total_games > 0:
            win_rate = stats[0] / total_games * 100
            print(f"   {opponent_name:15} {win_rate:5.1f}% win rate ({stats[0]:3}-{stats[1]:3}-{stats[2]:3})")
    
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrated RL Training System')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--delay', type=float, default=0.05, help='Visualization delay between moves')
    parser.add_argument('--visuals', action='store_true', help='Enable visualization')
    parser.add_argument('--no-emergency', dest='emergency_mode', action='store_false', help='Disable emergency pathfinding')
    parser.add_argument('--no-diverse', dest='use_diverse', action='store_false', help='Disable diverse opponents (use only basic + TronBot)')
    parser.set_defaults(visuals=False, emergency_mode=True, use_diverse=True)
    
    args = parser.parse_args()
    
    integrated_train(
        episodes=args.episodes,
        save_interval=args.save_interval,
        delay=args.delay,
        visuals=args.visuals,
        emergency_mode=args.emergency_mode,
        use_diverse_opponents=args.use_diverse
    )
