def generate_report(state, my_agent):
    board = state['board'][:]
    #Find my position
    if list(my_agent['my_position']) == state['agent1_trail'][-1]:
        my_x, my_y = state['agent1_trail'][-1]
        opp_x, opp_y = state['agent2_trail'][-1]
        for y in range(len(board)):
            for x in range(len(board[y])):
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
    




