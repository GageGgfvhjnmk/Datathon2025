"""
Local Judge - Simulates a full game between two agents in a single file
No need to run separate agent servers - both strategies run locally
"""

import os
import time
from collections import deque
from case_closed_game import Game, Direction, GameResult, EMPTY
import local_ram


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
        return local_ram.send_move_agent1(self.game)
    
    def send_move_agent2(self):
        return local_ram.send_move_agent1(self.game)
    
    def run_game(self, delay=0.1, visualize=True):
        """Run the full game simulation"""
        result = None
        
        while result is None:
            if visualize:
                self.visualize_board()
                time.sleep(delay)
            
            # Store previous positions to detect collision reasons
            agent1_prev_pos = self.game.agent1.trail[-1] if self.game.agent1.alive else None
            agent2_prev_pos = self.game.agent2.trail[-1] if self.game.agent2.alive else None
            
            # Get moves from both agents
            dir1, boost1 = self.send_move_agent1()
            dir2, boost2 = self.send_move_agent2()
            
            # Calculate next positions
            if agent1_prev_pos:
                dx1, dy1 = dir1.value
                agent1_next_pos = ((agent1_prev_pos[0] + dx1) % self.game.board.width,
                                   (agent1_prev_pos[1] + dy1) % self.game.board.height)
            else:
                agent1_next_pos = None
                
            if agent2_prev_pos:
                dx2, dy2 = dir2.value
                agent2_next_pos = ((agent2_prev_pos[0] + dx2) % self.game.board.width,
                                   (agent2_prev_pos[1] + dy2) % self.game.board.height)
            else:
                agent2_next_pos = None
            
            # Execute the turn
            result = self.game.step(dir1, dir2, boost1, boost2)
            
            # Determine specific death reason and calculate score
            if result is not None:
                score = 0
                death_reason = ""
                
                if result == GameResult.DRAW:
                    score = 0
                    death_reason = "Draw (both agents died or max turns)"
                    
                elif result == GameResult.AGENT1_WIN:
                    # Agent 2 died - determine why
                    if agent2_next_pos in self.game.agent2.trail:
                        # Agent 2 hit its own trail
                        score = 10
                        death_reason = "Agent 2 crossed Agent 2's path (+10)"
                    elif agent2_next_pos in self.game.agent1.trail:
                        # Agent 2 hit Agent 1's trail
                        score = 25
                        death_reason = "Agent 2 crossed Agent 1's path (+25)"
                    else:
                        score = 10  # Default if reason unclear
                        death_reason = "Agent 2 died (unclear reason, +10)"
                        
                elif result == GameResult.AGENT2_WIN:
                    # Agent 1 died - determine why
                    if agent1_next_pos in self.game.agent1.trail:
                        # Agent 1 hit its own trail
                        score = -25
                        death_reason = "Agent 1 crossed Agent 1's path (-25)"
                    elif agent1_next_pos in self.game.agent2.trail:
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
    print("Starting Local Judge - Rammer vs Evader")
    print("Press Ctrl+C to stop\n")
    
    judge = LocalJudge()
    result = judge.run_game(delay=0.05, visualize=True)
