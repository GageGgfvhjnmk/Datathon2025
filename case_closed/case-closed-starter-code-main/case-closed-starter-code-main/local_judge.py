"""
Local Judge - Simulates a full game between two agents in a single file
No need to run separate agent servers - both strategies run locally
Agent 1: Trained RL Agent (from integrated_train_rl.py with emergency pathfinding)
Agent 2: RL agent (heuristic-guided DQN)
"""

import os
import time
from collections import deque
from case_closed_game import Game, Direction, GameResult, EMPTY
import agent_strategies
import RL_agent_strategy


class LocalJudge:
    def __init__(self):
        self.game = Game()
        self.last_death_reason = ""
        
    def visualize_board(self):
        """Display the current game state with colored output"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        board = [[' ' for _ in range(self.game.board.width)] for _ in range(self.game.board.height)]
        
        # Mark agent 1 trail (green)
        for i, pos in enumerate(self.game.agent1.trail):
            x, y = pos
            if i == len(self.game.agent1.trail) - 1:  # Head
                board[y][x] = '\033[92m1\033[0m'
            else:
                board[y][x] = '\033[92m█\033[0m'
        
        # Mark agent 2 trail (red)
        for i, pos in enumerate(self.game.agent2.trail):
            x, y = pos
            if i == len(self.game.agent2.trail) - 1:  # Head
                board[y][x] = '\033[91m2\033[0m'
            else:
                board[y][x] = '\033[91m█\033[0m'
        
        # Print board
        print("=" * (self.game.board.width * 2 + 2))
        print(f"Turn: {self.game.turns} | Agent1: {self.game.agent1.length} trail, {self.game.agent1.boosts_remaining} boosts | Agent2: {self.game.agent2.length} trail, {self.game.agent2.boosts_remaining} boosts")
        print("=" * (self.game.board.width * 2 + 2))
        for row in board:
            print('|' + ''.join(cell if cell != ' ' else '.' for cell in row) + '|')
        print("=" * (self.game.board.width * 2 + 2))
    
    def send_move_agent1(self):
        """Agent 1: Trained RL Agent (from integrated_train_rl.py)"""
        return RL_agent_strategy.send_move_rl_agent(self.game, player_number=1)
    
    def send_move_agent2(self):
        """Agent 2: RL agent (heuristic-guided DQN)"""
        return RL_agent_strategy.send_move_agent2(self.game)
    
    def run_game(self, delay=0.1, visualize=True):
        """Run the full game simulation"""
        result = None
        agent1_will_hit_own = False
        agent1_will_hit_other = False
        agent2_will_hit_own = False
        agent2_will_hit_other = False
        agent1_invalid_move = False
        agent2_invalid_move = False
        
        while result is None:
            if visualize:
                self.visualize_board()
                time.sleep(delay)
            
            # Get moves from both agents
            dir1, boost1 = self.send_move_agent1()
            dir2, boost2 = self.send_move_agent2()
            
            # Check for invalid backwards moves BEFORE executing
            cur_dx1, cur_dy1 = self.game.agent1.direction.value
            req_dx1, req_dy1 = dir1.value
            agent1_invalid_move = (req_dx1, req_dy1) == (-cur_dx1, -cur_dy1)
            
            cur_dx2, cur_dy2 = self.game.agent2.direction.value
            req_dx2, req_dy2 = dir2.value
            agent2_invalid_move = (req_dx2, req_dy2) == (-cur_dx2, -cur_dy2)
            
            # Calculate next positions BEFORE the move to determine collision reasons
            agent1_head = self.game.agent1.trail[-1]
            agent2_head = self.game.agent2.trail[-1]
            
            # Calculate where each agent will move (accounting for boost)
            num_moves_1 = 2 if (boost1 and self.game.agent1.boosts_remaining > 0) else 1
            num_moves_2 = 2 if (boost2 and self.game.agent2.boosts_remaining > 0) else 1
            
            # Calculate final positions after all moves
            agent1_next_pos = agent1_head
            for _ in range(num_moves_1):
                dx1, dy1 = dir1.value
                agent1_next_pos = ((agent1_next_pos[0] + dx1) % self.game.board.width,
                                   (agent1_next_pos[1] + dy1) % self.game.board.height)
            
            agent2_next_pos = agent2_head
            for _ in range(num_moves_2):
                dx2, dy2 = dir2.value
                agent2_next_pos = ((agent2_next_pos[0] + dx2) % self.game.board.width,
                                   (agent2_next_pos[1] + dy2) % self.game.board.height)
            
            # Check collision reasons BEFORE executing the move
            agent1_will_hit_own = agent1_next_pos in self.game.agent1.trail
            agent1_will_hit_other = agent1_next_pos in self.game.agent2.trail
            agent2_will_hit_own = agent2_next_pos in self.game.agent2.trail
            agent2_will_hit_other = agent2_next_pos in self.game.agent1.trail
            
            # Execute the turn
            result = self.game.step(dir1, dir2, boost1, boost2)
        
        # After the loop, determine specific death reason and calculate score
        score = 0
        death_reason = ""
        
        if result == GameResult.DRAW:
            score = 0
            death_reason = "Draw (both agents died or max turns)"
            
        elif result == GameResult.AGENT1_WIN:
            # Agent 2 died - determine why
            if agent2_invalid_move:
                # Agent 2 made invalid backwards move
                score = 25
                death_reason = "Agent 2 made invalid move (backwards) (+25)"
            elif agent2_will_hit_own:
                # Agent 2 hit its own trail
                score = 10
                death_reason = "Agent 2 crossed Agent 2's path (+10)"
            elif agent2_will_hit_other:
                # Agent 2 hit Agent 1's trail
                score = 25
                death_reason = "Agent 2 crossed Agent 1's path (+25)"
            else:
                score = 10  # Default if reason unclear
                death_reason = "Agent 2 died (unclear reason, +10)"
                
        elif result == GameResult.AGENT2_WIN:
            # Agent 1 died - determine why
            if agent1_invalid_move:
                # Agent 1 made invalid backwards move
                score = -25
                death_reason = "Agent 1 made invalid move (backwards) (-25)"
            elif agent1_will_hit_own:
                # Agent 1 hit its own trail
                score = -25
                death_reason = "Agent 1 crossed Agent 1's path (-25)"
            elif agent1_will_hit_other:
                # Agent 1 hit Agent 2's trail
                score = -10
                death_reason = "Agent 1 crossed Agent 2's path (-10)"
            else:
                score = -10  # Default if reason unclear
                death_reason = "Agent 1 died (unclear reason, -10)"
        
        # Store death reason for statistics
        self.last_death_reason = death_reason
        
        # Show final board
        if visualize:
            self.visualize_board()
        
        # Print result
        print("\n" + "=" * 50)
        if result == GameResult.AGENT1_WIN:
            print("🎉 AGENT 1 (RAMMER) WINS!")
        elif result == GameResult.AGENT2_WIN:
            print("🎉 AGENT 2 (EVADER) WINS!")
        else:
            print("🤝 DRAW!")
        print("=" * 50)
        
        return result, score


def run_multiple_games(num_games=100, visualize_all=False, visualize_last=True, delay=0.01):
    """
    Run multiple games and generate a summary
    
    Args:
        num_games: Number of games to run
        visualize_all: Whether to visualize all games
        visualize_last: Whether to visualize the last game (ignored if visualize_all is True)
        delay: Delay between moves (for visualization)
    
    Returns:
        Dictionary with summary statistics
    """
    print("=" * 70)
    print(f"🎮 RUNNING {num_games} GAMES - TRAINED RL AGENT vs RL AGENT")
    print("=" * 70)
    print("Agent 1: Trained RL Agent (from integrated_train_rl.py)")
    print("         Features: Emergency pathfinding, trap detection, hard-coded safety")
    print("Agent 2: RL Agent (Heuristic-Guided DQN)")
    if visualize_all:
        print(f"🎬 Visualization: ENABLED for all games (delay: {delay}s)")
    elif visualize_last:
        print(f"🎬 Visualization: ENABLED for last game only")
    else:
        print(f"🎬 Visualization: DISABLED")
    print("=" * 70)
    print()
    
    # Statistics tracking
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_score = 0
    
    # Detailed win reasons
    agent1_win_reasons = {
        'invalid_move': 0,
        'hit_own_trail': 0,
        'hit_other_trail': 0,
        'other': 0
    }
    
    agent2_win_reasons = {
        'invalid_move': 0,
        'hit_own_trail': 0,
        'hit_other_trail': 0,
        'other': 0
    }
    
    # Run all games
    for game_num in range(1, num_games + 1):
        # Determine if this game should be visualized
        if visualize_all:
            should_visualize = True
        elif visualize_last:
            should_visualize = (game_num == num_games)
        else:
            should_visualize = False
        
        judge = LocalJudge()
        result, score = judge.run_game(delay=delay, visualize=should_visualize)
        
        # Track results
        total_score += score
        
        if result == GameResult.AGENT1_WIN:
            agent1_wins += 1
            if score == 25:
                # Agent 2 made invalid move or hit Agent 1's trail
                if "invalid" in judge.last_death_reason.lower():
                    agent1_win_reasons['invalid_move'] += 1
                else:
                    agent1_win_reasons['hit_other_trail'] += 1
            elif score == 10:
                # Agent 2 hit own trail
                agent1_win_reasons['hit_own_trail'] += 1
            else:
                agent1_win_reasons['other'] += 1
        elif result == GameResult.AGENT2_WIN:
            agent2_wins += 1
            if score == -25:
                # Agent 1 made invalid move or hit own trail
                if "invalid" in judge.last_death_reason.lower():
                    agent2_win_reasons['invalid_move'] += 1
                else:
                    agent2_win_reasons['hit_own_trail'] += 1
            elif score == -10:
                # Agent 1 hit Agent 2's trail
                agent2_win_reasons['hit_other_trail'] += 1
            else:
                agent2_win_reasons['other'] += 1
        else:
            draws += 1
        
        # Print progress every 10 games
        if game_num % 10 == 0 or game_num == num_games:
            win_rate_a1 = (agent1_wins / game_num) * 100
            win_rate_a2 = (agent2_wins / game_num) * 100
            draw_rate = (draws / game_num) * 100
            avg_score = total_score / game_num
            
            print(f"Game {game_num:3}/{num_games} | "
                  f"Agent1: {agent1_wins:3}W ({win_rate_a1:5.1f}%) | "
                  f"Agent2: {agent2_wins:3}W ({win_rate_a2:5.1f}%) | "
                  f"Draws: {draws:3} ({draw_rate:5.1f}%) | "
                  f"Avg Score: {avg_score:+6.2f}")
    
    # Generate summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTotal Games: {num_games}")
    print(f"\n🏆 OVERALL RESULTS:")
    print(f"   Agent 1 Wins: {agent1_wins:4} ({(agent1_wins/num_games)*100:5.1f}%)")
    print(f"   Agent 2 Wins: {agent2_wins:4} ({(agent2_wins/num_games)*100:5.1f}%)")
    print(f"   Draws:        {draws:4} ({(draws/num_games)*100:5.1f}%)")
    print(f"\n💯 TOTAL SCORE: {total_score:+.1f} (Average: {total_score/num_games:+.2f} per game)")
    
    # Win reason breakdown
    if agent1_wins > 0:
        print(f"\n📋 AGENT 1 WIN BREAKDOWN:")
        print(f"   Opponent Invalid Move:     {agent1_win_reasons['invalid_move']:4} ({(agent1_win_reasons['invalid_move']/agent1_wins)*100:5.1f}%)")
        print(f"   Opponent Hit Own Trail:    {agent1_win_reasons['hit_own_trail']:4} ({(agent1_win_reasons['hit_own_trail']/agent1_wins)*100:5.1f}%)")
        print(f"   Opponent Hit Agent1 Trail: {agent1_win_reasons['hit_other_trail']:4} ({(agent1_win_reasons['hit_other_trail']/agent1_wins)*100:5.1f}%)")
        if agent1_win_reasons['other'] > 0:
            print(f"   Other:                     {agent1_win_reasons['other']:4} ({(agent1_win_reasons['other']/agent1_wins)*100:5.1f}%)")
    
    if agent2_wins > 0:
        print(f"\n📋 AGENT 2 WIN BREAKDOWN:")
        print(f"   Opponent Invalid Move:     {agent2_win_reasons['invalid_move']:4} ({(agent2_win_reasons['invalid_move']/agent2_wins)*100:5.1f}%)")
        print(f"   Opponent Hit Own Trail:    {agent2_win_reasons['hit_own_trail']:4} ({(agent2_win_reasons['hit_own_trail']/agent2_wins)*100:5.1f}%)")
        print(f"   Opponent Hit Agent2 Trail: {agent2_win_reasons['hit_other_trail']:4} ({(agent2_win_reasons['hit_other_trail']/agent2_wins)*100:5.1f}%)")
        if agent2_win_reasons['other'] > 0:
            print(f"   Other:                     {agent2_win_reasons['other']:4} ({(agent2_win_reasons['other']/agent2_wins)*100:5.1f}%)")
    
    print("\n" + "=" * 70)
    
    # Determine overall winner
    if agent1_wins > agent2_wins:
        print("🏆 AGENT 1 DOMINATES!")
    elif agent2_wins > agent1_wins:
        print("🏆 AGENT 2 DOMINATES!")
    else:
        print("🤝 PERFECTLY BALANCED!")
    print("=" * 70)
    
    return {
        'total_games': num_games,
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'total_score': total_score,
        'avg_score': total_score / num_games,
        'agent1_win_reasons': agent1_win_reasons,
        'agent2_win_reasons': agent2_win_reasons
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Judge - Run games between RL agents')
    parser.add_argument('--games', type=int, default=100, help='Number of games to run (default: 100)')
    parser.add_argument('--visualize-all', action='store_true', help='Visualize all games')
    parser.add_argument('--visualize-last', action='store_true', help='Visualize only the last game')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between moves for visualization (default: 0.01s)')
    parser.add_argument('--single', action='store_true', help='Run a single visualized game')
    
    args = parser.parse_args()
    
    if args.single:
        # Run single game with visualization
        print("Starting Local Judge - Trained RL Agent vs RL Agent")
        print("Agent 1: Trained RL Agent (from integrated_train_rl.py)")
        print("         Features: Emergency pathfinding, trap detection, hard-coded safety")
        print("Agent 2: RL Agent (Heuristic-Guided DQN)")
        print("Press Ctrl+C to stop\n")
        
        judge = LocalJudge()
        result, score = judge.run_game(delay=0.05, visualize=True)
    else:
        # Run multiple games
        run_multiple_games(
            num_games=args.games, 
            visualize_all=args.visualize_all,
            visualize_last=args.visualize_last, 
            delay=args.delay
        )

