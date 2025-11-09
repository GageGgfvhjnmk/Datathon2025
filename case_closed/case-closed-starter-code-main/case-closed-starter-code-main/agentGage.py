from typing import Tuple, List, Optional
import agent
from case_closed_game import Direction, EMPTY
import numpy as np


def gage_logic(player_number):
    game = agent.GLOBAL_GAME
    board = game.board

    my_agent = game.agent1 if player_number == 1 else game.agent2
    other = game.agent2 if player_number == 1 else game.agent1
    
    print("Pass 1")

    head = my_agent.trail[-1]
    lowest = 10000
    lowest_dir = my_agent.direction  # fallback

    print("pass 2")

    grid_np = np.array(board.grid)  # kept to minimize edits; not required for the loop fill
    w, h = board.width, board.height

    for dir_enum in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
        ddx, ddy = dir_enum.value
        pos = board._torus_check((head[0] + ddx, head[1] + ddy))

        print("pass 3")
        if other and other.alive and pos in (other.trail + my_agent.trail):
            continue

        # center point of the window, wrapped
        cx, cy = board._torus_check((head[0] + 4 * ddx, head[1] + 4 * ddy))

        # CHANGED: build a 6x6 NumPy matrix by filling it via a wrapped for loop (y first, then x)
        matrix = np.empty((6, 6), dtype=np.float64)
        for r, oy in enumerate(range(-3, 3)):
            for c, ox in enumerate(range(-3, 3)):
                x, y = board._torus_check((cx + ox, cy + oy))
                matrix[r, c] = float(board.grid[y][x])

        print("m")
        # matrix is always 6x6 now, so no empty check needed; keeping prints as-is
        for row in matrix:
            for j in row:
                print(j, end="")
            print()

        avg = np.mean(matrix)
        print("pass 4, avg =", avg)

        if avg < lowest:
            lowest = avg
            lowest_dir = dir_enum
    
    move = lowest_dir.name
    print("Return Value:", move)
    return move


gage_logic(1)
        
'''
#-------------------------

# gage_logic.py

def gage_logic(player_number):

    game = agent.GLOBAL_GAME
    board = game.board

    my_agent = game.agent1 if player_number == 1 else game.agent2
    other = game.agent2 if player_number == 1 else game.agent1
    
    print("Pass 1")


    head = my_agent.trail[-1]
    direction = my_agent.direction
    dx, dy = direction.value
    lowest = 10000
    lowest_dir = direction

    print("pass 2")
    
    for dir in [(0,1), (1,0), (0,-1), (-1,0)]:
        # if dir == -direction:
        #     continue
        pos = game.board._torus_check((head[0] + dx, head[1] + dy))

        print("pass 3")
        if other and other.alive and pos in (other.trail + my_agent.trail):
            continue

        cx, cy = (head[0] + 4*dx, head[1] + 4*dy)
        matrix = game.board.grid[cx-3 : cx+3][cy-3 : cy+3]
        print("m")
        for i in matrix:
            for j in i:
                print(j, end="")
            print()

        avg = np.mean(np.array(matrix))
        print("pass 4")
        if lowest > avg:
            lowest = avg
            lowest_dir = dir
    
    move = Direction(lowest_dir).name


    print("Return Value: ", move)

    return move


'''
