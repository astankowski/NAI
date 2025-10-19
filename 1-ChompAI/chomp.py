"""
Chomp (Terminal Game)

Chomp is a two-player strategy game played on a grid of squares (like a chocolate bar).
Players take turns choosing a square to eat — this removes that square and all squares
below and to the right. The top-left square is poisoned. Whoever eats it loses.

For detailed rules and strategies, visit:
https://en.wikipedia.org/wiki/Chomp

Authors:
- Aleksander Stankowski (s27549)
- Daniel Bieliński (s27292)

Environment Setup:
1. Clone the repository:
    git clone https://github.com/astankowski/chompAI.git
    cd chompAI

2. Create and activate a virtual environment:
   python3 -m venv .venv

   For macOS/Linux:
        source .venv/bin/activate
   For Windows:
        .\venv\Scripts\activate

3. Install dependencies:
    python3 -m pip install easyAI

4. Run the game:
    For chomp with AI player:
        python3 chompAI.py 
    For chomp without AI, player vs player:
        python3 chomp.py
"""

def create_board(rows, cols) -> list[list[bool]]:
    """
    Creates a game board by filling it with True values.

    Parameters:
        rows (int): number of rows
        cols (int): number of cols

    Returns:
        list[list[bool]]: Two dimensional array rows x cols filled with True values.
    """
    return [[True for _ in range(cols)] for _ in range(rows)]

def print_board(board):
    """
    Prints the game board along with row and column labels.

    Parameters:
        board: a two dimensional array with boolean values.
    """
    rows = len(board)
    cols = len(board[0])

    column_labels = ' ' + ''.join(str(c) for c in range(cols))
    print(column_labels)

    for r in range(rows):
        line = f"{r}"
        for c in range(cols):
            line += '¤' if board[r][c] else ' '
        print(line)

def valid_move(board, r, c) -> bool:
    """
    Check if the player move is valid. Chosen field has to be True and be within the board size.

    Parameters:
        r (int): row coordinate
        c (int): column coordinate
    
    Returns:
        bool: True if the move is valid, False if invalid.
    """
    rows = len(board)
    cols = len(board[0])
    return 0 <= r < rows and 0 <= c < cols and board[r][c]

def apply_move(board, r, c):
    """
    Apply the move on the board by replacing the value of the field with False.

    Parameters:
        r (int): row coordinate
        c (int): column coordinate
    """
    rows = len(board)
    cols = len(board[0])
    for i in range(r, rows):
        for j in range(c, cols):
            board[i][j] = False

def get_move(player, board):
    """
    Keep prompting user for a move until they enter a valid move in 'row col' format eg. 2 3

    Parameters:
        player (int): Value 1 or 2. Determines which player is currently making the move.
        board: a two dimensional array with boolean values.
    """
    while True:
        try:
            move = input(f"Player {player}, enter your move (row col): ")
            r, c = map(int, move.strip().split())
            if valid_move(board, r, c):
                return r, c
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Use: row col (e.g., 1 2)")

def is_game_over(board) -> bool:
    """
    Game over condition. If the [0][0] field has the value of False,
    this means that the poison has been eaten.

    Parameters:
        board: a two dimensional array with boolean values.

    Returns:
        bool: True if the poison has been eaten, False if poison not yet eaten.
    """
    return not board[0][0]

def chomp_game(rows=9, cols=9):
    """
    Runs the main Chomp game loop.

    Parameters:
        rows (int): number of rows in the board
        cols (int): number of columns in the board
    """

    board = create_board(rows, cols)
    current_player = 1

    while True:
        print_board(board)

        if is_game_over(board):
            print(f"Player {3 - current_player} ate the poison. Player {current_player} wins")
            break

        r, c = get_move(current_player, board)
        apply_move(board, r, c)

        current_player = 3 - current_player

if __name__ == '__main__':
    chomp_game()
