"""
Script for training the chess policy network using imitation learning (behavioral cloning).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added this import
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from common.utils import square_indices_to_move  # For potential debugging/evaluation
from policy_network import ChessPolicy


class ChessDataset(Dataset):
    def __init__(self, states_file, actions_file):
        self.states = np.load(states_file)
        self.actions = np.load(actions_file)

        # Ensure states are in (N, C, H, W) format for PyTorch CNNs
        # Original: (N, H, W, C) -> Transpose to (N, C, H, W)
        self.states = np.transpose(self.states, (0, 3, 1, 2))

        assert len(self.states) == len(
            self.actions
        ), "States and actions must have the same number of samples."

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.from_numpy(self.states[idx]).float()
        start_square = torch.tensor(self.actions[idx][0], dtype=torch.long)
        end_square = torch.tensor(self.actions[idx][1], dtype=torch.long)
        return state, start_square, end_square


def train_imitation_learning(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs=10,
    device="cpu",
    model_save_path="models/imitation_learning/pretrained_policy.pth",
):
    """
    Trains the policy network using behavioral cloning.

    Args:
        model (nn.Module): The ChessPolicy network.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        device (str): Device to train on ('cpu' or 'cuda').
        model_save_path (str): Path to save the trained model.
    """
    model.to(device)

    # Loss functions for start and end squares
    start_criterion = nn.CrossEntropyLoss()
    end_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_start_correct = 0
        train_end_correct = 0
        total_samples = 0

        for states, start_squares, end_squares in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"
        ):
            states, start_squares, end_squares = (
                states.to(device),
                start_squares.to(device),
                end_squares.to(device),
            )

            optimizer.zero_grad()

            # Forward pass for start square prediction
            start_probs, _ = model(states)
            start_loss = start_criterion(start_probs, start_squares)

            # Prepare one-hot encoded start squares for end square prediction
            start_squares_one_hot = F.one_hot(start_squares, num_classes=64).float()

            # Forward pass for end square prediction
            _, end_probs = model(states, start_squares_one_hot)
            end_loss = end_criterion(end_probs, end_squares)

            # Total loss
            loss = start_loss + end_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * states.size(0)

            # Calculate accuracy
            _, predicted_start = torch.max(start_probs.data, 1)
            train_start_correct += (predicted_start == start_squares).sum().item()

            _, predicted_end = torch.max(end_probs.data, 1)
            train_end_correct += (predicted_end == end_squares).sum().item()

            total_samples += states.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_start_acc = train_start_correct / total_samples
        epoch_train_end_acc = train_end_correct / total_samples

        print(
            f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f}, Start Acc: {epoch_train_start_acc:.4f}, End Acc: {epoch_train_end_acc:.4f}"
        )

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        val_start_correct = 0
        val_end_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for states, start_squares, end_squares in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"
            ):
                states, start_squares, end_squares = (
                    states.to(device),
                    start_squares.to(device),
                    end_squares.to(device),
                )

                start_probs, _ = model(states)
                start_loss = start_criterion(start_probs, start_squares)

                start_squares_one_hot = F.one_hot(start_squares, num_classes=64).float()
                _, end_probs = model(states, start_squares_one_hot)
                end_loss = end_criterion(end_probs, end_squares)

                loss = start_loss + end_loss

                val_running_loss += loss.item() * states.size(0)

                _, predicted_start = torch.max(start_probs.data, 1)
                val_start_correct += (predicted_start == start_squares).sum().item()

                _, predicted_end = torch.max(end_probs.data, 1)
                val_end_correct += (predicted_end == end_squares).sum().item()

                val_total_samples += states.size(0)

        epoch_val_loss = val_running_loss / val_total_samples
        epoch_val_start_acc = val_start_correct / val_total_samples
        epoch_val_end_acc = val_end_correct / val_total_samples

        print(
            f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f}, Start Acc: {epoch_val_start_acc:.4f}, End Acc: {epoch_val_end_acc:.4f}"
        )

        # Save the model if validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(
                f"Model saved to {model_save_path} with validation loss: {best_val_loss:.4f}"
            )

    print("Imitation learning training complete.")


def main():
    # Define paths to your processed data
    states_file = "data/processed/states.npy"
    actions_file = "data/processed/actions.npy"

    # Check if data files exist
    if not os.path.exists(states_file) or not os.path.exists(actions_file):
        print(f"Error: Data files not found. Please run data_preparation.py first.")
        sys.exit(1)

    # Create dataset
    full_dataset = ChessDataset(states_file, actions_file)

    # Split into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
    )

    # Initialize model, optimizer, and device
    model = ChessPolicy()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Train the model
    train_imitation_learning(
        model, train_loader, val_loader, optimizer, epochs=10, device=device
    )


if __name__ == "__main__":
    main()
