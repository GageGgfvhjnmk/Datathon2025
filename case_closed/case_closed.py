import os
import time

arena = [['0' for _ in range(18)] for _ in range(20)]
character_1 = 'A'
character_2 = 'B'
a_trail = []
b_trail = []

def print_arena():
    os.system('cls')
    for row in arena:
        print(' '.join(row))

print_arena()
#Assuming A travels linearly from bottom right to top left
for i in range(17, -1, -1):
    # print(f"A's turn to move to position ({i}, {i})")
    arena[i][i] = f'\033[91m{character_1}\033[0m'  # Red color for A
    a_trail.append((i, i))
    print_arena()
    # Simulate delay
    time.sleep(0.1)
