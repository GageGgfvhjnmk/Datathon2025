"""
Local Judge - Simulates a full game between two agents in a single file
No need to run separate agent servers - both strategies run locally
"""

import os
import time
from collections import deque
from case_closed_game import Game, Direction, GameResult, EMPTY


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
        """
        Agent 1's strategy - Rammer with safety and backwards prevention
        """
        my_agent = self.game.agent1
        opponent = self.game.agent2
        my_pos = my_agent.trail[-1]
        opponent_pos = opponent.trail[-1]
        board = self.game.board
        
        def get_direction_to_target(my_pos, target_pos, board, my_trail):
            x1, y1 = my_pos
            x2, y2 = target_pos
            
            # Determine current direction (to avoid moving backwards)
            current_direction = None
            if len(my_trail) >= 2:
                prev_pos = my_trail[-2]
                dx = (x1 - prev_pos[0]) % board.width
                dy = (y1 - prev_pos[1]) % board.height
                
                # Normalize for torus wrapping
                if dx == board.width - 1:
                    dx = -1
                if dy == board.height - 1:
                    dy = -1
                
                if dx == 1 and dy == 0:
                    current_direction = "RIGHT"
                elif dx == -1 and dy == 0:
                    current_direction = "LEFT"
                elif dx == 0 and dy == 1:
                    current_direction = "DOWN"
                elif dx == 0 and dy == -1:
                    current_direction = "UP"
            
            # Define opposite directions
            opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            backwards = opposite.get(current_direction) if current_direction else None
            
            # Calculate directional distances with torus wrapping
            dx_right = (x2 - x1) % board.width
            dx_left = (x1 - x2) % board.width
            dy_down = (y2 - y1) % board.height
            dy_up = (y1 - y2) % board.height
            
            # Create a list of (direction, distance, next_pos) tuples
            direction_data = [
                ("RIGHT", dx_right, ((x1 + 1) % board.width, y1)),
                ("LEFT", dx_left, ((x1 - 1) % board.width, y1)),
                ("DOWN", dy_down, (x1, (y1 + 1) % board.height)),
                ("UP", dy_up, (x1, (y1 - 1) % board.height))
            ]
            
            # Filter out backwards move
            if backwards:
                direction_data = [(d, dist, pos) for d, dist, pos in direction_data if d != backwards]
            
            # Check safety for each move and filter to only safe moves
            safe_moves = []
            unsafe_moves = []
            
            for direction, distance, next_pos in direction_data:
                # Check if position is safe (not on a trail, except opponent's head)
                is_safe = (board.get_cell_state(next_pos) == EMPTY) or (next_pos == target_pos)
                
                if is_safe:
                    safe_moves.append((direction, distance))
                else:
                    unsafe_moves.append((direction, distance))
            
            # Sort safe moves by distance (shortest first)
            safe_moves.sort(key=lambda x: x[1])
            
            # Return the safest move with shortest distance
            if safe_moves:
                return safe_moves[0][0]
            elif unsafe_moves:
                unsafe_moves.sort(key=lambda x: x[1])
                return unsafe_moves[0][0]
            else:
                return "UP"
        
        move_str = get_direction_to_target(my_pos, opponent_pos, board, my_agent.trail)
        
        # Convert to Direction enum
        direction_map = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT
        }
        
        # Calculate distance for boost decision
        dx = min(abs(opponent_pos[0] - my_pos[0]), board.width - abs(opponent_pos[0] - my_pos[0]))
        dy = min(abs(opponent_pos[1] - my_pos[1]), board.height - abs(opponent_pos[1] - my_pos[1]))
        distance = dx + dy
        
        use_boost = my_agent.boosts_remaining > 0 and distance < 5
        
        return direction_map[move_str], use_boost
    
    def send_move_agent2(self):
        """
        Agent 2's strategy - Simple survival strategy (stays away from opponent)
        """
        my_agent = self.game.agent2
        opponent = self.game.agent1
        my_pos = my_agent.trail[-1]
        opponent_pos = opponent.trail[-1]
        board = self.game.board
        
        def get_direction_away_from_target(my_pos, target_pos, board, my_trail):
            x1, y1 = my_pos
            x2, y2 = target_pos
            
            # Determine current direction
            current_direction = None
            if len(my_trail) >= 2:
                prev_pos = my_trail[-2]
                dx = (x1 - prev_pos[0]) % board.width
                dy = (y1 - prev_pos[1]) % board.height
                
                if dx == board.width - 1:
                    dx = -1
                if dy == board.height - 1:
                    dy = -1
                
                if dx == 1 and dy == 0:
                    current_direction = "RIGHT"
                elif dx == -1 and dy == 0:
                    current_direction = "LEFT"
                elif dx == 0 and dy == 1:
                    current_direction = "DOWN"
                elif dx == 0 and dy == -1:
                    current_direction = "UP"
            
            opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            backwards = opposite.get(current_direction) if current_direction else None
            
            # Calculate directional distances (to MAXIMIZE distance from opponent)
            dx_right = (x2 - x1) % board.width
            dx_left = (x1 - x2) % board.width
            dy_down = (y2 - y1) % board.height
            dy_up = (y1 - y2) % board.height
            
            direction_data = [
                ("RIGHT", dx_left, ((x1 + 1) % board.width, y1)),  # Going right increases left distance
                ("LEFT", dx_right, ((x1 - 1) % board.width, y1)),  # Going left increases right distance
                ("DOWN", dy_up, (x1, (y1 + 1) % board.height)),
                ("UP", dy_down, (x1, (y1 - 1) % board.height))
            ]
            
            if backwards:
                direction_data = [(d, dist, pos) for d, dist, pos in direction_data if d != backwards]
            
            safe_moves = []
            unsafe_moves = []
            
            for direction, distance, next_pos in direction_data:
                is_safe = (board.get_cell_state(next_pos) == EMPTY) or (next_pos == target_pos)
                
                if is_safe:
                    safe_moves.append((direction, distance))
                else:
                    unsafe_moves.append((direction, distance))
            
            # Sort safe moves by distance (LONGEST first - maximize distance)
            safe_moves.sort(key=lambda x: x[1], reverse=True)
            
            if safe_moves:
                return safe_moves[0][0]
            elif unsafe_moves:
                unsafe_moves.sort(key=lambda x: x[1], reverse=True)
                return unsafe_moves[0][0]
            else:
                return "UP"
        
        move_str = get_direction_away_from_target(my_pos, opponent_pos, board, my_agent.trail)
        
        direction_map = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT
        }
        
        use_boost = False  # Agent 2 doesn't use boosts
        
        return direction_map[move_str], use_boost
    
    def run_game(self, delay=0.1, visualize=True):
        """Run the full game simulation"""
        result = None
        
        while result is None:
            if visualize:
                self.visualize_board()
                time.sleep(delay)
            
            # Get moves from both agents
            dir1, boost1 = self.send_move_agent1()
            dir2, boost2 = self.send_move_agent2()
            
            # Execute the turn
            result = self.game.step(dir1, dir2, boost1, boost2)
        
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
        print("=" * 50)
        pr
        
        return result


if __name__ == "__main__":
    print("Starting Local Judge - Rammer vs Evader")
    print("Press Ctrl+C to stop\n")
    
    judge = LocalJudge()
    result = judge.run_game(delay=0.05, visualize=True)
