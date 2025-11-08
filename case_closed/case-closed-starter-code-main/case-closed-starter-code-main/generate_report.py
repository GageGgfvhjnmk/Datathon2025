state = {'board': [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'agent1_trail': [[1, 2], [2, 2], [2, 1], [2, 0], [2, 17], [2, 16], [2, 15], [3, 15], [4, 15], [5, 15]], 'agent2_trail': [[17, 15], [16, 15], [15, 15], [14, 15], [13, 15], [12, 15], [11, 15], [10, 15], [9, 15], [8, 15]], 'agent1_length': 10, 'agent2_length': 10, 'agent1_alive': True, 'agent2_alive': True, 'agent1_boosts': 3, 'agent2_boosts': 3, 'turn_count': 8, 'player_number': 1}
my_agent = {'my_position': (5, 15), 'my_trail': [(1, 2), (2, 2), (2, 1), (2, 0), (2, 17), (2, 16), (2, 15), (3, 15), (4, 15), (5, 15)], 'my_length': 10, 'my_boosts': 3}

def generate_report():
    board = state['board'][:]
    #Find my position
    if list(my_agent['my_position']) == state['agent1_trail'][-1]:
        my_x, my_y = state['agent1_trail'][-1]
        opp_x, opp_y = state['agent2_trail'][-1]
        for y in range(len(board)):
            for x in range(len(board[y])):
                print((x,y))
                if (x, y) == (my_x, my_y):
                    board[y][x] = '\033[92mA\033[0m'  # Mark my trail on the board for visualization
                elif (x, y) == (opp_x, opp_y):
                    board[y][x] = '\033[91mB\033[0m'  # Mark opponent's trail on the board for visualization
                elif [x,y] in state['agent1_trail'][:-1]:
                    board[y][x] = '\033[92m█\033[0m'  # Mark my trail on the board for visualization
                elif [x,y] in state['agent2_trail'][:-1]:
                    board[y][x] = '\033[91m█\033[0m'  # Mark opponent's trail on the board for visualization
                else:
                    board[y][x] = '\033[90m█\033[0m'
        
        # in state['agent1_trail'][:-1]:
        #     x, y = cell
        #     board[y][x] = '\033[92m█\033[0m'  # Mark my trail on the board for visualization
        # for cell in state['agent2_trail'][:-1]:
        #     x, y = cell
        #     board[y][x] = '\033[91m█\033[0m'  # Mark opponent's trail on the board for visualization
        # for cell in 
    else:
        my_x, my_y = state['agent2_trail'][-1]
        opp_x, opp_y = state['agent1_trail'][-1]
        for cell in state['agent2_trail'][:-1]:
            x, y = cell
            board[y][x] = '\033[92m█\033[0m'  # Mark my trail on the board for visualization
        for cell in state['agent1_trail'][:-1]:
            x, y = cell
            board[y][x] = '\033[91m█\033[0m'  # Mark opponent's trail on the board for visualization
    print(f"My Position: {my_agent['my_position']}, Opponent Position: {(opp_x, opp_y)}")
    board[my_y][my_x] = f'\033[92mA\033[0m'  # Mark my position on the board for visualization
    board[opp_y][opp_x] = f'\033[91mB\033[0m'  # Mark my position on the board for visualization
    print("Board State:")
    for row in board:
        string = ''.join(str(cell) for cell in row)
        print(string)
    

generate_report()




