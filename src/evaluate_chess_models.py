import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
import time 

QUEEN_DIR_OFFSETS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)
] 
QUEEN_LIKE_MOVES_PLANES = 56

KNIGHT_MOVES_OFFSETS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
] 
KNIGHT_MOVES_PLANES = 8 

PAWN_PROMOTION_TYPES = [
    (0, 1),   
    (-1, 1),  
    (1, 1)    
] 
PROMOTION_PIECES_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP] 
PAWN_PROMOTION_PLANES = 9 

TOTAL_ACTION_PLANES = QUEEN_LIKE_MOVES_PLANES + KNIGHT_MOVES_PLANES + PAWN_PROMOTION_PLANES 
POLICY_OUTPUT_SIZE = TOTAL_ACTION_PLANES * 64 

NUM_HISTORY_FRAMES = 8 
PLANES_PER_FRAME = 14  
AUX_PLANES = 7         
NUM_PLANES = (NUM_HISTORY_FRAMES * PLANES_PER_FRAME) + AUX_PLANES 

def encode_board_lc0(board: chess.Board, history: deque) -> np.ndarray:
    
    encoded_state = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    piece_plane_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    all_boards = list(history) + [board]
    all_boards = all_boards[-NUM_HISTORY_FRAMES:]

    for i, historical_board in enumerate(reversed(all_boards)):
        frame_offset = i * PLANES_PER_FRAME

        for sq in chess.SQUARES:
            piece = historical_board.piece_at(sq)
            if piece:
                plane_idx = piece_plane_map[piece.piece_type]
                if piece.color != historical_board.turn:
                    plane_idx += 6 

                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                encoded_state[frame_offset + plane_idx, rank, file] = 1

        encoded_state[frame_offset + 12, :, :] = 1 if historical_board.turn == historical_board.turn else 0

        if historical_board.ep_square is not None:
            rank = chess.square_rank(historical_board.ep_square)
            file = chess.square_file(historical_board.ep_square)
            encoded_state[frame_offset + 13, rank, file] = 1

    aux_offset = NUM_HISTORY_FRAMES * PLANES_PER_FRAME

    if board.has_kingside_castling_rights(chess.WHITE):
        encoded_state[aux_offset + 0, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded_state[aux_offset + 1, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded_state[aux_offset + 2, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded_state[aux_offset + 3, :, :] = 1

    encoded_state[aux_offset + 4, :, :] = board.halfmove_clock / 100.0

    encoded_state[aux_offset + 5, :, :] = board.fullmove_number / 200.0

    if board.is_repetition():
        encoded_state[aux_offset + 6, :, :] = 1

    if board.turn == chess.BLACK:
        encoded_state = np.rot90(encoded_state, k=2, axes=(1, 2)).copy()

        for i in range(NUM_HISTORY_FRAMES):
            frame_offset = i * PLANES_PER_FRAME
            for j in range(6): 
                temp = np.copy(encoded_state[frame_offset + j, :, :]) 
                encoded_state[frame_offset + j, :, :] = encoded_state[frame_offset + j + 6, :, :] 
                encoded_state[frame_offset + j + 6, :, :] = temp

        temp = np.copy(encoded_state[aux_offset + 0, :, :]) 
        encoded_state[aux_offset + 0, :, :] = encoded_state[aux_offset + 2, :, :] 
        encoded_state[aux_offset + 2, :, :] = temp

        temp = np.copy(encoded_state[aux_offset + 1, :, :]) 
        encoded_state[aux_offset + 1, :, :] = encoded_state[aux_offset + 3, :, :] 
        encoded_state[aux_offset + 3, :, :] = temp

    return encoded_state

def get_action_plane_idx_lc0(move: chess.Move, board: chess.Board) -> int:
    
    from_sq = move.from_square
    to_sq = move.to_square

    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    to_rank, to_file = chess.square_rank(to_sq), chess.square_file(to_sq)

    dx = to_file - from_file 
    dy = to_rank - from_rank 

    if move.promotion:
        try:
            promotion_piece_idx = PROMOTION_PIECES_ORDER.index(move.promotion)
        except ValueError:
            if move.promotion == chess.KNIGHT:
                promotion_piece_idx = PROMOTION_PIECES_ORDER.index(chess.QUEEN)
            else:
                raise ValueError(f"Unexpected promotion piece: {move.promotion} in move {move.uci()}")

        if dx == 0: 
            pawn_move_type_idx = 0
        elif dx == -1: 
            pawn_move_type_idx = 1
        elif dx == 1: 
            pawn_move_type_idx = 2
        else:
            raise ValueError(f"Invalid pawn promotion dx: {dx} for move {move.uci()}")

        action_plane_idx = QUEEN_LIKE_MOVES_PLANES + KNIGHT_MOVES_PLANES + \
                           (promotion_piece_idx * 3) + pawn_move_type_idx
        return action_plane_idx

    knight_move_delta = (dx, dy)
    if knight_move_delta in KNIGHT_MOVES_OFFSETS:
        action_plane_idx = QUEEN_LIKE_MOVES_PLANES + KNIGHT_MOVES_OFFSETS.index(knight_move_delta)
        return action_plane_idx

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if (abs_dx == 0 and abs_dy > 0) or \
       (abs_dy == 0 and abs_dx > 0) or \
       (abs_dx == abs_dy and abs_dx > 0):

        distance = max(abs_dx, abs_dy)
        if distance == 0 or distance > 7:
            raise ValueError(f"Invalid distance {distance} for ray move: {move.uci()}")

        norm_dx = dx // distance if distance > 0 else 0
        norm_dy = dy // distance if distance > 0 else 0

        try:
            direction_idx = QUEEN_DIR_OFFSETS.index((norm_dx, norm_dy))
        except ValueError:
            raise ValueError(f"Invalid direction for ray move: {move.uci()} (dx={dx}, dy={dy})")

        action_plane_idx = direction_idx * 7 + (distance - 1)
        return action_plane_idx

    raise ValueError(f"Move {move.uci()} from {board.fen()} does not fit Lc0 action plane scheme.")

def encode_move_lc0_style(move: chess.Move, board: chess.Board) -> int:
    
    from_sq_encoded = move.from_square
    if board.turn == chess.BLACK:
        from_sq_encoded = chess.square_mirror(from_sq_encoded) 

    action_plane_idx = get_action_plane_idx_lc0(move, board)

    return from_sq_encoded * TOTAL_ACTION_PLANES + action_plane_idx

def decode_move_lc0_style(predicted_idx: int, board: chess.Board) -> chess.Move:
    
    from_sq_pred = predicted_idx // TOTAL_ACTION_PLANES
    action_plane_idx_pred = predicted_idx % TOTAL_ACTION_PLANES

    if board.turn == chess.BLACK:
        from_sq_pred = chess.square_mirror(from_sq_pred)

    for legal_move in board.legal_moves:
        try:
            encoded_legal_move_idx = encode_move_lc0_style(legal_move, board)
            if encoded_legal_move_idx == predicted_idx:
                return legal_move
        except ValueError:
            pass

    if board.legal_moves:
        return random.choice(list(board.legal_moves))
    else:
        return None 

class ResidualBlock(nn.Module):
    
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
        out += identity 
        return F.relu(out)

class ChessNet(nn.Module):
    
    def __init__(self, num_residual_blocks=5, num_channels=128):
        super(ChessNet, self).__init__()
        self.conv_input = nn.Conv2d(NUM_PLANES, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_residual_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, TOTAL_ACTION_PLANES, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(TOTAL_ACTION_PLANES)

        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128) 
        self.value_fc2 = nn.Linear(128, 1) 

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        policy_output = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_output = policy_output.view(-1, POLICY_OUTPUT_SIZE) 

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * 8 * 8) 
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) 

        return policy_output, value 

def load_chess_model(path_to_pth: str, device: torch.device) -> ChessNet:
    
    model = ChessNet(num_residual_blocks=5, num_channels=128)
    model.load_state_dict(torch.load(path_to_pth, map_location=device))
    model.to(device)
    model.eval() 
    print(f"Model loaded successfully from {path_to_pth}")
    return model

class StockfishPlayer:
    
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
        
        if self.engine:
            self.engine.quit()
            print("Stockfish engine closed.")

def get_random_move(board: chess.Board) -> chess.Move:
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)

def get_dqn_move(model: ChessNet, board: chess.Board, history_deque: deque, device: torch.device) -> chess.Move:
    
    with torch.no_grad():
        encoded_state = encode_board_lc0(board, history_deque)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)

        q_values_all_actions, _ = model(state_tensor)

        masked_q_values = torch.full_like(q_values_all_actions, float('-inf'))
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        for move in legal_moves:
            try:
                encoded_move_idx = encode_move_lc0_style(move, board)
                masked_q_values[0, encoded_move_idx] = q_values_all_actions[0, encoded_move_idx]
            except ValueError:
                continue

        predicted_action_idx = torch.argmax(masked_q_values).item()
        chosen_move = decode_move_lc0_style(predicted_action_idx, board)

        if chosen_move is None or chosen_move not in legal_moves:
            return random.choice(legal_moves)

        return chosen_move

def get_imitation_move(model: ChessNet, board: chess.Board, history_deque: deque, device: torch.device) -> chess.Move:
    
    with torch.no_grad():
        encoded_state = encode_board_lc0(board, history_deque)
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(device)

        policy_logits, _ = model(state_tensor)

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
                continue 
        if best_move is None:
            return random.choice(legal_moves)

        return best_move

def play_game(player1_func, player1_model_or_engine, player2_func, player2_model_or_engine, device: torch.device, max_moves: int = 200):
    
    board = chess.Board()
    history_deque_p1 = deque([board.copy() for _ in range(NUM_HISTORY_FRAMES - 1)], maxlen=NUM_HISTORY_FRAMES - 1)
    history_deque_p2 = deque([board.copy() for _ in range(NUM_HISTORY_FRAMES - 1)], maxlen=NUM_HISTORY_FRAMES - 1)

    moves_made = 0
    while not board.is_game_over() and moves_made < max_moves:
        current_player_color = board.turn

        if current_player_color == chess.WHITE:
            if player1_func == get_dqn_move or player1_func == get_imitation_move:
                move = player1_func(player1_model_or_engine, board, history_deque_p1, device)
            else: 
                move = player1_func(board)
        else:
            if player2_func == get_dqn_move or player2_func == get_imitation_move:
                move = player2_func(player2_model_or_engine, board, history_deque_p2, device)
            else: 
                move = player2_func(board)

        if move is None:
            print(f"Warning: Player {current_player_color} failed to make a move. Game aborted.")
            return -1 if current_player_color == chess.WHITE else 1 

        board.push(move)
        history_deque_p1.append(board.copy())
        history_deque_p2.append(board.copy())
        moves_made += 1

    result = board.result()
    if result == "1-0": 
        return 1
    elif result == "0-1": 
        return -1
    else: 
        return 0

if __name__ == "__main__":
    DQN_MODEL_PATH = "models/dqn/dqn_chess_agent_final.pth"
    IM_MODEL_PATH = "models/imitation_learning/chess_model_lc0_imitation.pth"
    STOCKFISH_PATH = "/Users/nakulnarang/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
    NUM_GAMES_PER_MATCHUP = 50 

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

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
            if game_result == -1: 
                results["DQN_vs_Stockfish"]["wins"] += 1
            elif game_result == 1: 
                results["DQN_vs_Stockfish"]["losses"] += 1
            else:
                results["DQN_vs_Stockfish"]["draws"] += 1
        print("\n")

        print(f"\n--- Evaluating DQN Agent against Random Agent for {NUM_GAMES_PER_MATCHUP} games ---")
        for i in range(NUM_GAMES_PER_MATCHUP):
            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (DQN as White vs Random)...", end='\r')
            game_result = play_game(get_dqn_move, dqn_model, get_random_move, None, device) 
            if game_result == 1:
                results["DQN_vs_Random"]["wins"] += 1
            elif game_result == -1:
                results["DQN_vs_Random"]["losses"] += 1
            else:
                results["DQN_vs_Random"]["draws"] += 1

            print(f"  Game {i+1}/{NUM_GAMES_PER_MATCHUP} (DQN as Black vs Random)...", end='\r')
            game_result = play_game(get_random_move, None, get_dqn_move, dqn_model, device)
            if game_result == -1: 
                results["DQN_vs_Random"]["wins"] += 1
            elif game_result == 1: 
                results["DQN_vs_Random"]["losses"] += 1
            else:
                results["DQN_vs_Random"]["draws"] += 1
        print("\n") 


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
            if game_result == -1: 
                results["IM_vs_Stockfish"]["wins"] += 1
            elif game_result == 1: 
                results["IM_vs_Stockfish"]["losses"] += 1
            else:
                results["IM_vs_Stockfish"]["draws"] += 1
        print("\n")
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
            if game_result == -1: 
                results["IM_vs_Random"]["wins"] += 1
            elif game_result == 1: 
                results["IM_vs_Random"]["losses"] += 1
            else:
                results["IM_vs_Random"]["draws"] += 1
        print("\n") 

    stockfish_player.close()

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
