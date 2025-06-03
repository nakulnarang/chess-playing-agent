import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
import time # For potential delays in Stockfish communication

# --- Lc0-Style Move Encoding Constants ---
QUEEN_DIR_OFFSETS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)
] # N, NE, E, SE, S, SW, W, NW - 8 directions for queen-like moves
QUEEN_LIKE_MOVES_PLANES = 56 # 8 directions * 7 distances (max distance on an 8x8 board)

KNIGHT_MOVES_OFFSETS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
] # 8 possible knight moves
KNIGHT_MOVES_PLANES = 8 # Total: 56 + 8 = 64 action planes so far

PAWN_PROMOTION_TYPES = [
    (0, 1),   # Straight push (e.g., e7-e8)
    (-1, 1),  # Diagonal Left Capture (e.g., b7-a8)
    (1, 1)    # Diagonal Right Capture (e.g., b7-c8)
] # Relative moves for pawn promotions
PROMOTION_PIECES_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP] # Order of promotion pieces (Knight promotions are handled specially)
PAWN_PROMOTION_PLANES = 9 # 3 pawn move types * 3 promotion piece types

TOTAL_ACTION_PLANES = QUEEN_LIKE_MOVES_PLANES + KNIGHT_MOVES_PLANES + PAWN_PROMOTION_PLANES # 56 + 8 + 9 = 73
POLICY_OUTPUT_SIZE = TOTAL_ACTION_PLANES * 64 # 73 * 64 = 4672 (Total number of possible encoded moves)

NUM_HISTORY_FRAMES = 8 # Number of past board states to include (7 in deque + current board)
PLANES_PER_FRAME = 14  # 6 piece types * 2 colors + 1 for side to move + 1 for en passant
AUX_PLANES = 7         # Castling rights (4), halfmove clock (1), fullmove number (1), repetition (1)
NUM_PLANES = (NUM_HISTORY_FRAMES * PLANES_PER_FRAME) + AUX_PLANES # Total planes for the input state (119)

# --- Lc0-Style Board Encoding ---
def encode_board_lc0(board: chess.Board, history: deque) -> np.ndarray:
    """
    Encodes a chess.Board object into a multi-plane numpy array,
    including historical context, similar to Lc0's input.
    The output is a (NUM_PLANES, 8, 8) numpy array.
    """
    encoded_state = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    piece_plane_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    # Combine historical boards with the current board
    all_boards = list(history) + [board]
    # Ensure we only take the last NUM_HISTORY_FRAMES boards
    all_boards = all_boards[-NUM_HISTORY_FRAMES:]

    # Iterate through historical boards (from most recent to oldest)
    for i, historical_board in enumerate(reversed(all_boards)):
        frame_offset = i * PLANES_PER_FRAME

        # Encode pieces on the board
        for sq in chess.SQUARES:
            piece = historical_board.piece_at(sq)
            if piece:
                plane_idx = piece_plane_map[piece.piece_type]
                # If it's the opponent's piece in that frame, use the offset for opponent's pieces
                if piece.color != historical_board.turn:
                    plane_idx += 6 # Offset for opponent's pieces (planes 6-11)

                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                encoded_state[frame_offset + plane_idx, rank, file] = 1

        # Side to move plane (1 if it's the side to move for *that specific historical board*, 0 otherwise)
        encoded_state[frame_offset + 12, :, :] = 1 if historical_board.turn == historical_board.turn else 0

        # En passant target square plane
        if historical_board.ep_square is not None:
            rank = chess.square_rank(historical_board.ep_square)
            file = chess.square_file(historical_board.ep_square)
            encoded_state[frame_offset + 13, rank, file] = 1

    # Encode auxiliary features for the *current* board state
    aux_offset = NUM_HISTORY_FRAMES * PLANES_PER_FRAME

    # Castling rights (4 planes: White K-side, White Q-side, Black K-side, Black Q-side)
    if board.has_kingside_castling_rights(chess.WHITE):
        encoded_state[aux_offset + 0, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded_state[aux_offset + 1, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded_state[aux_offset + 2, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded_state[aux_offset + 3, :, :] = 1

    # Half-move clock (normalized to 0-1 range)
    encoded_state[aux_offset + 4, :, :] = board.halfmove_clock / 100.0

    # Full-move number (normalized to 0-1 range)
    encoded_state[aux_offset + 5, :, :] = board.fullmove_number / 200.0

    # Repetition flag (1 if current position is 2x or 3x repetition)
    if board.is_repetition():
        encoded_state[aux_offset + 6, :, :] = 1

    # --- Symmetrize Board for Current Player (AlphaZero/Lc0 Style) ---
    # If it's Black's turn, flip the board 180 degrees and swap piece planes
    # so the input always looks like White's perspective to the network.
    if board.turn == chess.BLACK:
        # Flip each plane 180 degrees (rotate 90 degrees twice)
        encoded_state = np.rot90(encoded_state, k=2, axes=(1, 2)).copy()

        # Swap piece planes within each frame (player's pieces <-> opponent's pieces)
        for i in range(NUM_HISTORY_FRAMES):
            frame_offset = i * PLANES_PER_FRAME
            for j in range(6): # P, N, B, R, Q, K
                temp = np.copy(encoded_state[frame_offset + j, :, :]) # Player's piece plane
                encoded_state[frame_offset + j, :, :] = encoded_state[frame_offset + j + 6, :, :] # Opponent's piece plane
                encoded_state[frame_offset + j + 6, :, :] = temp

        # Swap castling rights (White K-side with Black K-side, etc.)
        temp = np.copy(encoded_state[aux_offset + 0, :, :]) # White King-side
        encoded_state[aux_offset + 0, :, :] = encoded_state[aux_offset + 2, :, :] # Black King-side
        encoded_state[aux_offset + 2, :, :] = temp

        temp = np.copy(encoded_state[aux_offset + 1, :, :]) # White Queen-side
        encoded_state[aux_offset + 1, :, :] = encoded_state[aux_offset + 3, :, :] # Black Queen-side
        encoded_state[aux_offset + 3, :, :] = temp

    return encoded_state

def get_action_plane_idx_lc0(move: chess.Move, board: chess.Board) -> int:
    """
    Determines the action plane index (0-72) for a given chess.Move based on Lc0's scheme.
    This function *assumes* the move is legal for the given board state.
    It handles board symmetry for black moves by mirroring the squares first.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Mirror squares if it's Black's turn to maintain a consistent perspective
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)

    dx = to_file - from_file # Change in file
    dy = to_rank - from_rank # Change in rank

    # 1. Pawn Promotions (Planes 64-72)
    if move.promotion:
        try:
            # Get index of the promotion piece (Queen, Rook, Bishop)
            promotion_piece_idx = PROMOTION_PIECES_ORDER.index(move.promotion)
        except ValueError:
            # Lc0's 9-plane promotion scheme often maps knight promotions to queen promotions
            if move.promotion == chess.KNIGHT:
                promotion_piece_idx = PROMOTION_PIECES_ORDER.index(chess.QUEEN)
            else:
                raise ValueError(f"Unexpected promotion piece: {move.promotion} in move {move.uci()}")

        # Determine the pawn move type (straight, diagonal left, diagonal right)
        if dx == 0: # Straight push
            pawn_move_type_idx = 0
        elif dx == -1: # Diagonal Left Capture
            pawn_move_type_idx = 1
        elif dx == 1: # Diagonal Right Capture
            pawn_move_type_idx = 2
        else:
            raise ValueError(f"Invalid pawn promotion dx: {dx} for move {move.uci()}")

        # Calculate the final action plane index for promotion
        action_plane_idx = QUEEN_LIKE_MOVES_PLANES + KNIGHT_MOVES_PLANES + \
                           (promotion_piece_idx * 3) + pawn_move_type_idx
        return action_plane_idx

    # 2. Knight Moves (Planes 56-63)
    knight_move_delta = (dx, dy)
    if knight_move_delta in KNIGHT_MOVES_OFFSETS:
        action_plane_idx = QUEEN_LIKE_MOVES_PLANES + KNIGHT_MOVES_OFFSETS.index(knight_move_delta)
        return action_plane_idx

    # 3. Queen-like Moves (Planes 0-55) - Handles ALL non-promotion, non-knight moves (rooks, bishops, queens, kings, non-promotion pawns)
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Check if it's a straight (rank/file) or diagonal move
    if (abs_dx == 0 and abs_dy > 0) or \
       (abs_dy == 0 and abs_dx > 0) or \
       (abs_dx == abs_dy and abs_dx > 0):

        distance = max(abs_dx, abs_dy)
        if distance == 0 or distance > 7:
            raise ValueError(f"Invalid distance {distance} for ray move: {move.uci()}")

        # Normalize dx, dy to get the direction vector (e.g., (0,1) for North)
        norm_dx = dx // distance if distance > 0 else 0
        norm_dy = dy // distance if distance > 0 else 0

        try:
            direction_idx = QUEEN_DIR_OFFSETS.index((norm_dx, norm_dy))
        except ValueError:
            raise ValueError(f"Invalid direction for ray move: {move.uci()} (dx={dx}, dy={dy})")

        # Calculate the final action plane index for queen-like moves
        action_plane_idx = direction_idx * 7 + (distance - 1)
        return action_plane_idx

    raise ValueError(f"Move {move.uci()} from {board.fen()} does not fit Lc0 action plane scheme.")

def encode_move_lc0_style(move: chess.Move, board: chess.Board) -> int:
    """
    Encodes a chess.Move object into a single integer index (0-4671)
    using the Lc0 style encoding (from_square * TOTAL_ACTION_PLANES + action_plane_idx).
    Handles board symmetry for black moves by mirroring the from_square.
    """
    from_sq_encoded = move.from_square
    if board.turn == chess.BLACK:
        from_sq_encoded = chess.square_mirror(from_sq_encoded) # Mirror from_square for black's perspective

    action_plane_idx = get_action_plane_idx_lc0(move, board)

    return from_sq_encoded * TOTAL_ACTION_PLANES + action_plane_idx

def decode_move_lc0_style(predicted_idx: int, board: chess.Board) -> chess.Move:
    """
    Decodes an Lc0-style integer index back to a chess.Move object given the current board.
    This function iterates through all legal moves on the board and attempts to match
    the predicted index with the encoded form of a legal move.
    If no direct match is found, it falls back to a random legal move.
    """
    # Extract the predicted 'from' square and action plane index
    from_sq_pred = predicted_idx // TOTAL_ACTION_PLANES
    action_plane_idx_pred = predicted_idx % TOTAL_ACTION_PLANES

    # Mirror the 'from' square back if it was originally a black move
    if board.turn == chess.BLACK:
        from_sq_pred = chess.square_mirror(from_sq_pred)

    # Iterate through legal moves to find a match
    for legal_move in board.legal_moves:
        try:
            # Encode each legal move and compare with the predicted index
            encoded_legal_move_idx = encode_move_lc0_style(legal_move, board)
            if encoded_legal_move_idx == predicted_idx:
                return legal_move
        except ValueError:
            # This can happen if a legal move's encoding is not perfectly handled
            pass

    # Fallback: If the predicted_idx does not correspond to a legal move,
    # choose a random legal move. This is crucial for robustness in RL.
    # print(f"Warning: No direct legal move found for predicted index {predicted_idx} on board {board.fen()}. Returning random legal move.")
    if board.legal_moves:
        return random.choice(list(board.legal_moves))
    else:
        return None # No legal moves available (e.g., game over)

# --- Neural Network Architecture (PyTorch) ---
class ResidualBlock(nn.Module):
    """
    A standard Residual Block used in AlphaZero/Lc0 style networks.
    Consists of two convolutional layers with Batch Normalization and ReLU activations,
    plus a skip connection.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Skip connection
        return F.relu(out)

class ChessNet(nn.Module):
    """
    The neural network architecture, adapted from the original notebook's ChessNet.
    It outputs both policy logits (or Q-values for DQN) and a value prediction.
    """
    def __init__(self, num_residual_blocks=5, num_channels=128):
        super(ChessNet, self).__init__()
        # Initial Convolutional Block: Takes the encoded board state as input
        self.conv_input = nn.Conv2d(NUM_PLANES, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Stack of Residual Blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_residual_blocks)])

        # Policy Head (outputs logits for actions or Q-values for DQN)
        self.policy_conv = nn.Conv2d(num_channels, TOTAL_ACTION_PLANES, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(TOTAL_ACTION_PLANES)

        # Value Head: Predicts the game outcome (-1 to 1)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128) # Flattened 8x8 feature map to 128 nodes
        self.value_fc2 = nn.Linear(128, 1) # Output a single scalar value

    def forward(self, x):
        # Input Block
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual Blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        policy_output = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_output = policy_output.view(-1, POLICY_OUTPUT_SIZE) # Flatten to (batch_size, 4672)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * 8 * 8) # Flatten the 1x8x8 feature map
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # Use tanh to squash output to -1 to 1 range

        return policy_output, value # Return policy_output (logits/Q-values) and value

# --- Model Loading Function ---
def load_chess_model(path_to_pth: str, device: torch.device) -> ChessNet:
    """
    Loads a ChessNet model from a .pth file.
    """
    model = ChessNet(num_residual_blocks=5, num_channels=128)
    model.load_state_dict(torch.load(path_to_pth, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    print(f"Model loaded successfully from {path_to_pth}")
    return model

# --- Stockfish Interaction Class ---
class StockfishPlayer:
    """
    A wrapper for the Stockfish chess engine.
    """
    def __init__(self, engine_path: str, skill_level: int = 1, time_limit: float = 0.1):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            self.engine.configure({"Skill Level": skill_level})
            self.time_limit = time_limit
            print(f"Stockfish engine initialized at {engine_path} with Skill Level {skill_level}")
        except Exception as e:
            print(f"Error initializing Stockfish engine: {e}")
            self.engine = None

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Gets the best move from Stockfish for the current board.
        """
        if not self.engine:
            return None
        try:
            result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
            return result.move
        except chess.engine.EngineError as e:
            print(f"Stockfish engine error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while getting Stockfish move: {e}")
            return None

    def close(self):
        """
        Closes the Stockfish engine.
        """
        if self.engine:
            self.engine.quit()
            print("Stockfish engine closed.")

# --- Random Agent Move Selection Function ---
def get_random_move(board: chess.Board) -> chess.Move:
    """
    Selects a random legal move from the current board.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)

# --- Agent Move Selection Functions ---
def get_dqn_move(model: ChessNet, board: chess.Board, history_deque: deque, device: torch.device) -> chess.Move:
    """
    Selects a move for the DQN agent.
    """
    with torch.no_grad():
        encoded_state = encode_board_lc0(board, history_deque)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)

        q_values_all_actions, _ = model(state_tensor)

        # Mask out illegal moves
        masked_q_values = torch.full_like(q_values_all_actions, float('-inf'))
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        for move in legal_moves:
            try:
                encoded_move_idx = encode_move_lc0_style(move, board)
                masked_q_values[0, encoded_move_idx] = q_values_all_actions[0, encoded_move_idx]
            except ValueError:
                # This can happen if a legal move's encoding is not perfectly handled.
                # The decode_move_lc0_style has a fallback, but here we just skip.
                continue

        predicted_action_idx = torch.argmax(masked_q_values).item()
        chosen_move = decode_move_lc0_style(predicted_action_idx, board)

        # Fallback if decode_move_lc0_style returns None or an illegal move (shouldn't happen with masking)
        if chosen_move is None or chosen_move not in legal_moves:
            # print(f"Warning: DQN agent chose an invalid move or decode failed. Falling back to random legal move.")
            return random.choice(legal_moves)

        return chosen_move

def get_imitation_move(model: ChessNet, board: chess.Board, history_deque: deque, device: torch.device) -> chess.Move:
    """
    Selects a move for the Imitation Learning agent.
    """
    with torch.no_grad():
        encoded_state = encode_board_lc0(board, history_deque)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)

        policy_logits, _ = model(state_tensor)

        # Filter legal moves and find the one with the highest logit
        best_move = None
        max_logit = float('-inf')
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        for move in legal_moves:
            try:
                encoded_move_idx = encode_move_lc0_style(move, board)
                current_logit = policy_logits[0, encoded_move_idx].item()
                if current_logit > max_logit:
                    max_logit = current_logit
                    best_move = move
            except ValueError:
                continue # Skip moves that cannot be encoded

        if best_move is None:
            # Fallback if no legal move could be encoded or found (should be rare)
            # print("Warning: Imitation agent could not find a valid move. Falling back to random legal move.")
            return random.choice(legal_moves)

        return best_move

# --- Game Simulation Function ---
def play_game(player1_func, player1_model_or_engine, player2_func, player2_model_or_engine, device: torch.device, max_moves: int = 200):
    """
    Simulates a single game of chess between two players.
    Returns:
        1 if player1 wins, -1 if player2 wins, 0 for draw.
    """
    board = chess.Board()
    # Initialize history deque for both players
    history_deque_p1 = deque([board.copy() for _ in range(NUM_HISTORY_FRAMES - 1)], maxlen=NUM_HISTORY_FRAMES - 1)
    history_deque_p2 = deque([board.copy() for _ in range(NUM_HISTORY_FRAMES - 1)], maxlen=NUM_HISTORY_FRAMES - 1)

    moves_made = 0
    while not board.is_game_over() and moves_made < max_moves:
        current_player_color = board.turn

        if current_player_color == chess.WHITE:
            # Player 1 (White) makes a move
            if player1_func == get_dqn_move or player1_func == get_imitation_move:
                move = player1_func(player1_model_or_engine, board, history_deque_p1, device)
            else: # Stockfish
                move = player1_func(board)
        else:
            # Player 2 (Black) makes a move
            if player2_func == get_dqn_move or player2_func == get_imitation_move:
                move = player2_func(player2_model_or_engine, board, history_deque_p2, device)
            else: # Stockfish
                move = player2_func(board)

        if move is None:
            # This can happen if an agent fails to produce a move or Stockfish errors out
            # Treat as a loss for the player who couldn't move
            print(f"Warning: Player {current_player_color} failed to make a move. Game aborted.")
            return -1 if current_player_color == chess.WHITE else 1 # Opponent wins

        board.push(move)
        # Update history for both players after the move is made
        history_deque_p1.append(board.copy())
        history_deque_p2.append(board.copy())
        moves_made += 1

    result = board.result()
    if result == "1-0": # White wins
        return 1
    elif result == "0-1": # Black wins
        return -1
    else: # Draw
        return 0

# --- Main Evaluation Loop ---
if __name__ == "__main__":
    # Configuration
    DQN_MODEL_PATH = "models/dqn/dqn_chess_agent_final.pth"
    IM_MODEL_PATH = "models/imitation_learning/chess_model_lc0_imitation.pth"
    STOCKFISH_PATH = "/Users/nakulnarang/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
    NUM_GAMES_PER_MATCHUP = 50 # Total games for each model vs Stockfish (half as white, half as black)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load models
    dqn_model = None
    im_model = None
    try:
        dqn_model = load_chess_model(DQN_MODEL_PATH, device)
    except Exception as e:
        print(f"Could not load DQN model from {DQN_MODEL_PATH}: {e}")

    try:
        im_model = load_chess_model(IM_MODEL_PATH, device)
    except Exception as e:
        print(f"Could not load Imitation Learning model from {IM_MODEL_PATH}: {e}")

    # Initialize Stockfish
    stockfish_player = StockfishPlayer(STOCKFISH_PATH, skill_level=1)
    if not stockfish_player.engine:
        print("Stockfish engine not available. Cannot proceed with evaluation.")
        exit()

    results = {
        "DQN_vs_Stockfish": {"wins": 0, "losses": 0, "draws": 0},
        "IM_vs_Stockfish": {"wins": 0, "losses": 0, "draws": 0},
        "DQN_vs_Random": {"wins": 0, "losses": 0, "draws": 0},
        "IM_vs_Random": {"wins": 0, "losses": 0, "draws": 0}
    }

    # Evaluate DQN Agent
    if dqn_model:
        print(f"\n--- Evaluating DQN Agent against Stockfish (Level 1) for {NUM_GAMES_PER_MATCHUP} games ---")
        for i in range(NUM_GAMES_PER_MATCHUP):
            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (DQN as White vs Stockfish)...", end='\r')
            game_result = play_game(get_dqn_move, dqn_model, stockfish_player.get_move, stockfish_player, device)
            if game_result == 1:
                results["DQN_vs_Stockfish"]["wins"] += 1
            elif game_result == -1:
                results["DQN_vs_Stockfish"]["losses"] += 1
            else:
                results["DQN_vs_Stockfish"]["draws"] += 1

            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (DQN as Black vs Stockfish)...", end='\r')
            game_result = play_game(stockfish_player.get_move, stockfish_player, get_dqn_move, dqn_model, device)
            if game_result == -1: # DQN wins as Black
                results["DQN_vs_Stockfish"]["wins"] += 1
            elif game_result == 1: # DQN loses as Black
                results["DQN_vs_Stockfish"]["losses"] += 1
            else:
                results["DQN_vs_Stockfish"]["draws"] += 1
        print("\n") # Newline after progress updates

        print(f"\n--- Evaluating DQN Agent against Random Agent for {NUM_GAMES_PER_MATCHUP} games ---")
        for i in range(NUM_GAMES_PER_MATCHUP):
            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (DQN as White vs Random)...", end='\r')
            game_result = play_game(get_dqn_move, dqn_model, get_random_move, None, device) # Random agent doesn't need a model/engine
            if game_result == 1:
                results["DQN_vs_Random"]["wins"] += 1
            elif game_result == -1:
                results["DQN_vs_Random"]["losses"] += 1
            else:
                results["DQN_vs_Random"]["draws"] += 1

            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (DQN as Black vs Random)...", end='\r')
            game_result = play_game(get_random_move, None, get_dqn_move, dqn_model, device)
            if game_result == -1: # DQN wins as Black
                results["DQN_vs_Random"]["wins"] += 1
            elif game_result == 1: # DQN loses as Black
                results["DQN_vs_Random"]["losses"] += 1
            else:
                results["DQN_vs_Random"]["draws"] += 1
        print("\n") # Newline after progress updates


    # Evaluate Imitation Learning Agent
    if im_model:
        print(f"\n--- Evaluating Imitation Learning Agent against Stockfish (Level 1) for {NUM_GAMES_PER_MATCHUP} games ---")
        for i in range(NUM_GAMES_PER_MATCHUP):
            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (IM as White vs Stockfish)...", end='\r')
            game_result = play_game(get_imitation_move, im_model, stockfish_player.get_move, stockfish_player, device)
            if game_result == 1:
                results["IM_vs_Stockfish"]["wins"] += 1
            elif game_result == -1:
                results["IM_vs_Stockfish"]["losses"] += 1
            else:
                results["IM_vs_Stockfish"]["draws"] += 1

            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (IM as Black vs Stockfish)...", end='\r')
            game_result = play_game(stockfish_player.get_move, stockfish_player, get_imitation_move, im_model, device)
            if game_result == -1: # IM wins as Black
                results["IM_vs_Stockfish"]["wins"] += 1
            elif game_result == 1: # IM loses as Black
                results["IM_vs_Stockfish"]["losses"] += 1
            else:
                results["IM_vs_Stockfish"]["draws"] += 1
        print("\n") # Newline after progress updates

        print(f"\n--- Evaluating Imitation Learning Agent against Random Agent for {NUM_GAMES_PER_MATCHUP} games ---")
        for i in range(NUM_GAMES_PER_MATCHUP):
            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (IM as White vs Random)...", end='\r')
            game_result = play_game(get_imitation_move, im_model, get_random_move, None, device)
            if game_result == 1:
                results["IM_vs_Random"]["wins"] += 1
            elif game_result == -1:
                results["IM_vs_Random"]["losses"] += 1
            else:
                results["IM_vs_Random"]["draws"] += 1

            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (IM as Black vs Random)...", end='\r')
            game_result = play_game(get_random_move, None, get_imitation_move, im_model, device)
            if game_result == -1: # IM wins as Black
                results["IM_vs_Random"]["wins"] += 1
            elif game_result == 1: # IM loses as Black
                results["IM_vs_Random"]["losses"] += 1
            else:
                results["IM_vs_Random"]["draws"] += 1
        print("\n") # Newline after progress updates

    # Close Stockfish
    stockfish_player.close()

    # Print Results
    print("\n--- Evaluation Results ---")
    for model_name, res in results.items():
        total_games = res["wins"] + res["losses"] + res["draws"]
        if total_games > 0:
            win_rate = (res["wins"] / total_games) * 100
            loss_rate = (res["losses"] / total_games) * 100
            draw_rate = (res["draws"] / total_games) * 100
            print(f"\n{model_name}:")
            print(f"  Total Games: {total_games}")
            print(f"  Wins: {res['wins']} ({win_rate:.2f}%)")
            print(f"  Losses: {res['losses']} ({loss_rate:.2f}%)")
            print(f"  Draws: {res['draws']} ({draw_rate:.2f}%)")
        else:
            print(f"\n{model_name}: No games played (model might not have loaded).")
