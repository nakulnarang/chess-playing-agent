"""
Script for downloading and processing chess game data from Lichess.
"""

import io
import os
import sys

import chess.pgn
import numpy as np
from tqdm import tqdm

from common.utils import move_to_square_indices, state_to_2d_array


def download_pgn(num_games=100, max_rating=2000):
    """
    Download PGN data from Lichess API.

    Args:
        num_games (int): Number of games to download
        max_rating (int): Maximum rating to filter games
    """
    url = f"https://lichess.org/api/games/user/DrNykterstein?max={num_games}&rated=true&perfType=bullet"
    pgn_dir = os.path.join("data", "pgn")
    output_file = os.path.join(pgn_dir, "sample_games.pgn")

    # Create directory if it doesn't exist
    os.makedirs(pgn_dir, exist_ok=True)

    if not os.path.exists(output_file):
        print(f"Downloading {num_games} games from Lichess API...")
        curl_command = (
            f'curl -v -H "Accept: application/x-chess-pgn" "{url}" -o "{output_file}"'
        )
        print(f"Executing command: {curl_command}")

        # Execute curl command
        result = os.system(curl_command)

        # Check if download was successful
        if result != 0:
            print("Error: Failed to download games")
            sys.exit(1)

        if not os.path.exists(output_file):
            print("Error: File was not created after download")
            sys.exit(1)

        # Check if file is empty
        if os.path.getsize(output_file) == 0:
            print("Error: Downloaded file is empty")
            os.remove(output_file)
            sys.exit(1)

        print("Download complete!")
        print(f"File size: {os.path.getsize(output_file)} bytes")

        # Print first few lines of the file
        print("\nFirst few lines of downloaded file:")
        with open(output_file, "r") as f:
            print(f.read(500))
    else:
        print(f"Sample games file already exists.")
        print(f"File size: {os.path.getsize(output_file)} bytes")

    return output_file, max_rating


def process_games(pgn_file, num_games=1000):
    """
    Process games from a PGN file and extract state-action pairs.

    Args:
        pgn_file (str): Path to the PGN file
        num_games (int): Number of games to process

    Returns:
        tuple: (states, actions) where states is a list of board states
               and actions is a list of (start_square, end_square) tuples
    """
    states = []
    actions = []

    print(f"Reading games from {pgn_file}")

    # Open PGN file
    with open(pgn_file, "r") as f:
        games_processed = 0
        for _ in tqdm(range(num_games), desc="Processing games"):
            game = chess.pgn.read_game(f)
            if game is None:  # End of file
                print("Reached end of file")
                break

            # Print game headers for debugging
            print(f"\nGame {games_processed + 1} headers:")
            for key, value in game.headers.items():
                print(f"{key}: {value}")

            print(f"\nProcessing game {games_processed + 1} moves...")

            board = game.board()
            for move in game.mainline_moves():
                # Store current state
                states.append(state_to_2d_array(board))

                # Store action (as start square, end square indices)
                actions.append(move_to_square_indices(move))

                # Make the move
                board.push(move)

            games_processed += 1

        print(f"\nProcessed {games_processed} games")

    return np.array(states), np.array(actions)


def save_dataset(states, actions, output_dir="data/processed"):
    """
    Save the processed states and actions as numpy arrays.

    Args:
        states (np.ndarray): Array of board states
        actions (np.ndarray): Array of actions
        output_dir (str): Directory to save the arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save arrays
    np.save(os.path.join(output_dir, "states.npy"), states)
    np.save(os.path.join(output_dir, "actions.npy"), actions)

    print(f"Saved {len(states)} state-action pairs to {output_dir}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")


def main():
    """Main function to download and process chess data."""
    # Download sample games
    pgn_file, _ = download_pgn(num_games=100)  # Start with 100 games

    # Process games
    states, actions = process_games(pgn_file, num_games=100)

    # Save processed data
    save_dataset(states, actions)

    print("\nData preparation completed!")
    print(f"You can find the processed data in the data/processed directory")


if __name__ == "__main__":
    main()
