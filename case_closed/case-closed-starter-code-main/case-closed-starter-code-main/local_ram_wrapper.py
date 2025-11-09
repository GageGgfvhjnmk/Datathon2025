"""
local_ram_wrapper.py

Wrapper for local_ram to provide both agent1 and agent2 functions with compatible interface.
"""
from case_closed_game import Direction
import local_ram

def send_move_agent1(game):
    """
    Wrapper for local_ram's send_move_agent1 with proper interface
    """
    return local_ram.send_move_agent1(game)

def send_move_agent2(game):
    """
    Create agent2 moves using the same logic as agent1 but adapted for player 2
    """
    # For agent2, we can reuse the same logic but swap the agent references
    # Create a temporary game state where agent2 is treated as agent1
    class TempGame:
        def __init__(self, original_game):
            self.board = original_game.board
            self.agent1 = original_game.agent2  # Swap agents
            self.agent2 = original_game.agent1  # Swap agents
    
    temp_game = TempGame(game)
    
    # Use the same logic but for the swapped agents
    direction, use_boost = local_ram.send_move_agent1(temp_game)
    return direction, use_boost