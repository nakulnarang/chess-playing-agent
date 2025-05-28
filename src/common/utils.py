"""
Utility functions for chess board representation and move encoding.
"""

import numpy as np
import chess

def state_to_2d_array(board):
    """
    Convert a chess.Board object to a multi-channel 2D array representation.
    
    The representation uses 15 channels:
    - Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - Channel 12: Castling rights (4 values for KQkq)
    - Channel 13: En passant target square
    - Channel 14: Side to move (1 for white, 0 for black)
    
    Args:
        board (chess.Board): The chess board to convert
        
    Returns:
        np.ndarray: 8x8x15 array representing the board state
    """
    # Initialize the state array
    state = np.zeros((8, 8, 15), dtype=np.float32)
    
    # Piece type to channel mapping (0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King)
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Fill piece channels
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = square // 8
            file = square % 8
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.WHITE:
                state[rank, file, channel] = 1
            else:
                state[rank, file, channel + 6] = 1
    
    # Fill castling rights (channel 12)
    if board.has_kingside_castling_rights(chess.WHITE):
        state[7, 4:8, 12] = 1  # White kingside
    if board.has_queenside_castling_rights(chess.WHITE):
        state[7, 0:5, 12] = 1  # White queenside
    if board.has_kingside_castling_rights(chess.BLACK):
        state[0, 4:8, 12] = 1  # Black kingside
    if board.has_queenside_castling_rights(chess.BLACK):
        state[0, 0:5, 12] = 1  # Black queenside
    
    # Fill en passant target square (channel 13)
    if board.ep_square is not None:
        rank = board.ep_square // 8
        file = board.ep_square % 8
        state[rank, file, 13] = 1
    
    # Fill side to move (channel 14)
    if board.turn == chess.WHITE:
        state[:, :, 14] = 1
    
    return state

def move_to_square_indices(move):
    """
    Convert a chess.Move to (start_square, end_square) indices.
    
    Args:
        move (chess.Move): The move to convert
        
    Returns:
        tuple: (start_square, end_square) where each square is an integer 0-63
    """
    return (move.from_square, move.to_square)

def square_indices_to_move(start_square, end_square):
    """
    Convert (start_square, end_square) indices to a chess.Move.
    
    Args:
        start_square (int): Starting square index (0-63)
        end_square (int): Ending square index (0-63)
        
    Returns:
        chess.Move: The corresponding chess move
    """
    return chess.Move(start_square, end_square)
