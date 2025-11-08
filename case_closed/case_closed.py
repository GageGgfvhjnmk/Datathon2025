arena = [['*' for _ in range(18)] for _ in range(20)]
character_1 = 'A'
character_2 = 'B'

def print_arena():
    for row in arena:
        print(' '.join(row))

print_arena()