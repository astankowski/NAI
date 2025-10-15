from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
from chomp import create_board, print_board, valid_move, apply_move, is_game_over

class ChompGame(TwoPlayerGame):
    """
    Chomp implemented with easyAI.
    """
    def __init__(self, players, rows=5, cols=7):
        self.players = players
        self.rows = rows
        self.cols = cols
        self.board = create_board(rows, cols)
        self.current_player = 1

    def possible_moves(self):
        """Return a list of valid moves as 'r c' strings."""
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                if valid_move(self.board, r, c):
                    moves.append(f"{r} {c}")
        return moves

    def make_move(self, move):
        """Apply the chosen move."""
        r, c = map(int, move.split(" "))
        apply_move(self.board, r, c)

    def is_over(self):
        """The game ends when poison has been eaten."""
        return is_game_over(self.board)

    def show(self):
        """Print the board for human players."""
        print_board(self.board)

    def win(self):
        """
        In Chomp, the player who eats poison loses.
        Since easyAI asks if the CURRENT PLAYER has won,
        if the poison is eaten now, it means they lost.
        So we return False in that case and True for the opponent.
        """
        
        return not self.board[0][0]

    def scoring(self):
        """Simple scoring function for the AI."""
        return 100 if self.win() else 0


if __name__ == '__main__':
    ai_algo = Negamax(6)  # search depth
    game = ChompGame([Human_Player(), AI_Player(ai_algo)])
    game.play()
