def generate_report(state, my_agent):
    board = state['board']
    
    # Determine positions
    if list(my_agent['my_position']) == state['agent1_trail'][-1]:
        my_x, my_y = state['agent1_trail'][-1]
        opp_x, opp_y = state['agent2_trail'][-1]
    else:
        my_x, my_y = state['agent2_trail'][-1]
        opp_x, opp_y = state['agent1_trail'][-1]
    
    print(f"My Position: {my_agent['my_position']}, Opponent Position: {(opp_x, opp_y)}")
    print("Board State (Raw Numerical):")
    
    # Print the raw numerical board
    for y, row in enumerate(board):
        row_str = ""
        for x, cell in enumerate(row):
            # Mark current positions with A/B, otherwise show the numeric value
            if (x, y) == (my_x, my_y):
                row_str += "A "
            elif (x, y) == (opp_x, opp_y):
                row_str += "B "
            else:
                row_str += f"{cell} "
        print(row_str)
    




