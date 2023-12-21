import os
import pickle
from typing import List
from ..errors import Errors
from .custom_types import Sample


class CheckpointManager:
    def __init__(self, checkpoint_folder="checkpoints"):
        """Initialize the CheckpointManager.

        Args:
            checkpoint_folder (str): The directory to store checkpoints and batch information.
        """
        self.checkpoint_folder = checkpoint_folder
        self.complete_folder = os.path.join(checkpoint_folder, "complete")
        self.remaining_folder = os.path.join(checkpoint_folder, "remaining")

        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.complete_folder, exist_ok=True)
        os.makedirs(self.remaining_folder, exist_ok=True)

    def save_checkpoint(self, check_point_extension: str, results_so_far: List[Sample]):
        """Save a checkpoint with partial results.

        Args:
            check_point_extension (str): Extension or identifier for the checkpoint.
            results_so_far (List[Sample]): Partial results to be saved.
        """
        file_path = os.path.join(
            self.complete_folder, f"checkpoint_{check_point_extension}.pkl"
        )
        with open(file_path, "wb") as file:
            pickle.dump(results_so_far, file)

    def save_all_batches(self, batches: dict):
        """Save all batches for potential recovery.

        Args:
            batches: Dictionary containing batches with batch numbers as keys and batches as values.
        """

        for i, batch in batches.items():
            checkpoint_path = os.path.join(
                self.remaining_folder, f"checkpoint_batch_{i}.pkl"
            )

            with open(checkpoint_path, "wb") as file:
                pickle.dump(batch, file)

    def load_checkpoint(self) -> List[Sample]:
        """Load complete checkpoints and concatenate results.

        Returns:
            List[Sample]: Concatenated list of results from all checkpoints.
        """
        concatenated_list = []

        for file_name in os.listdir(self.complete_folder):
            file_path = os.path.join(self.complete_folder, file_name)

            if file_name.endswith(".pkl"):
                with open(file_path, "rb") as file:
                    data_list = pickle.load(file)

                    concatenated_list.extend(data_list)

        return concatenated_list

    def load_remaining_batch(self) -> List[Sample]:
        """Load remaining batch checkpoints and concatenate results.

        Returns:
            List[Sample]: Concatenated list of results from all remaining batch checkpoints.
        """
        concatenated_list = []

        for file_name in os.listdir(self.remaining_folder):
            file_path = os.path.join(self.remaining_folder, file_name)

            if file_name.endswith(".pkl"):
                with open(file_path, "rb") as file:
                    data_list = pickle.load(file)

                    concatenated_list.extend(data_list)

        return concatenated_list
