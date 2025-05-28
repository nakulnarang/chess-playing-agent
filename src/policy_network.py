"""
Policy network for the chess agent using a two-headed architecture to predict start and end squares.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

class ChessPolicy(nn.Module):
    def __init__(self):
        super(ChessPolicy, self).__init__()
        
        # Input: 8x8x15 (board state with 15 channels)
        # CNN backbone
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Shared fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # Start square head
        self.start_fc = nn.Linear(512, 64)  # Output probabilities for 64 squares
        
        # End square head (includes concatenated start square info)
        self.end_fc1 = nn.Linear(512 + 64, 256)  # 512 from shared layers + 64 one-hot start square
        self.end_fc2 = nn.Linear(256, 64)  # Output probabilities for 64 squares
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def _shared_forward(self, x):
        """
        Forward pass through the shared layers of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 15, 8, 8)
            
        Returns:
            torch.Tensor: Output tensor from shared layers
        """
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return x

    def forward(self, x, start_square_one_hot=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 15, 8, 8)
            start_square_one_hot (torch.Tensor, optional): One-hot encoded start square for end square prediction
            
        Returns:
            tuple: (start_square_probs, end_square_probs)
        """
        shared_features = self._shared_forward(x)
        
        # Start square prediction
        start_logits = self.start_fc(shared_features)
        start_probs = F.softmax(start_logits, dim=1)
        
        # End square prediction (if start_square_one_hot is provided)
        if start_square_one_hot is not None:
            # Concatenate shared features with start square info
            combined = torch.cat([shared_features, start_square_one_hot], dim=1)
            end_hidden = F.relu(self.end_fc1(combined))
            end_logits = self.end_fc2(end_hidden)
            end_probs = F.softmax(end_logits, dim=1)
        else:
            end_probs = None
        
        return start_probs, end_probs

    def predict_move(self, board, device='cpu'):
        """
        Predict a move given a chess.Board object.
        
        Args:
            board (chess.Board): The current chess board state.
            device (str): Device to run inference on ('cpu' or 'cuda').
            
        Returns:
            chess.Move: The predicted legal move.
        """
        self.eval()  # Set to evaluation mode
        
        # Convert board to 2D array and then to tensor
        state_2d = torch.from_numpy(np.transpose(board_to_2d_array(board), (2, 0, 1))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. Predict start square probabilities
            start_probs, _ = self.forward(state_2d)
            start_probs = start_probs.squeeze(0) # Remove batch dimension
            
            # Create a mask for legal start squares
            legal_start_squares = set()
            for move in board.legal_moves:
                legal_start_squares.add(move.from_square)
            
            start_mask = torch.zeros_like(start_probs)
            for sq in legal_start_squares:
                start_mask[sq] = 1
            
            # Apply mask and re-normalize
            masked_start_probs = start_probs * start_mask
            if masked_start_probs.sum() == 0: # Fallback if no legal moves (shouldn't happen in normal play)
                return None
            masked_start_probs = masked_start_probs / masked_start_probs.sum()
            
            # Sample a start square
            start_square = torch.multinomial(masked_start_probs, 1).item()
            
            # 2. Predict end square probabilities conditioned on the chosen start square
            start_square_one_hot = F.one_hot(torch.tensor(start_square), num_classes=64).float().unsqueeze(0).to(device)
            _, end_probs = self.forward(state_2d, start_square_one_hot)
            end_probs = end_probs.squeeze(0) # Remove batch dimension
            
            # Create a mask for legal end squares given the chosen start square
            legal_end_squares_from_start = set()
            for move in board.legal_moves:
                if move.from_square == start_square:
                    legal_end_squares_from_start.add(move.to_square)
            
            end_mask = torch.zeros_like(end_probs)
            for sq in legal_end_squares_from_start:
                end_mask[sq] = 1
            
            # Apply mask and re-normalize
            masked_end_probs = end_probs * end_mask
            if masked_end_probs.sum() == 0: # Fallback if no legal end moves from this start square
                # This can happen if the sampled start_square was valid but had no legal moves
                # In this case, we should resample the start_square or pick a random legal move
                # For simplicity, let's just pick a random legal move if this happens
                print("Warning: Sampled start square has no legal moves. Picking a random legal move.")
                return random.choice(list(board.legal_moves))
            
            masked_end_probs = masked_end_probs / masked_end_probs.sum()
            
            # Sample an end square
            end_square = torch.multinomial(masked_end_probs, 1).item()
            
            # Construct the move
            predicted_move = chess.Move(start_square, end_square)
            
            # Final check for legality (should be legal due to masking)
            if predicted_move not in board.legal_moves:
                print(f"Warning: Predicted move {predicted_move.uci()} is not legal. Falling back to random legal move.")
                return random.choice(list(board.legal_moves))
            
            return predicted_move

# Helper function to convert board to 2D array (assuming it's in common.utils)
# This is a placeholder, ensure common.utils.state_to_2d_array is correctly imported and used.
from common.utils import state_to_2d_array as board_to_2d_array
import random # Import random for fallback
