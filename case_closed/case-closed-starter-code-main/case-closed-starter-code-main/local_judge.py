"""
Local Judge - Simulates a full game between two agents in a single file
No need to run separate agent servers - both strategies run locally
Agent 1: Hyper-optimized heuristic
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
        """Agent 1: Hyper-optimized heuristic"""
        return agent_strategies.send_move_agent1(self.game)
    
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
        print(f"Final Scores - Agent1: {self.game.agent1.length} trail | Agent2: {self.game.agent2.length} trail")
        print(f"Death Reason: {death_reason}")
        print(f"Score: {score}")
        print("=" * 50)
        
        return score


if __name__ == "__main__":
    print("Starting Local Judge - Heuristic Agent vs RL Agent (Heuristic-Guided DQN)")
    print("Agent 1: Hyper-optimized heuristic algorithm")
    print("Agent 2: Reinforcement Learning (learning from gameplay)")
    print("Press Ctrl+C to stop\n")
    
    judge = LocalJudge()
    result = judge.run_game(delay=0.05, visualize=True)
    print("Press Ctrl+C to stop\n")
    
    judge = LocalJudge()
    result = judge.run_game(delay=0.01, visualize=True)
