import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from collections import OrderedDict


def manual_normalize_feature(data: np.ndarray):
    """
    Normalize each feature independently to the range [0, 1] using min-max scaling.

    Args:
        data (np.ndarray): Input array of shape (samples, features).

    Returns:
        normalized_data (np.ndarray): Normalized array of same shape.
        min_vals (np.ndarray): Minimum values per feature.
        max_vals (np.ndarray): Maximum values per feature.
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)  # epsilon avoids division by zero
    return normalized_data, min_vals, max_vals


def manual_normalize_global(data: np.ndarray):
    """
    Normalize the entire dataset globally to the range [0, 1] using min-max scaling.

    Args:
        data (np.ndarray): Input array of shape (samples,).

    Returns:
        normalized_data (np.ndarray): Normalized array of same shape.
        min_val (float): Minimum value in the dataset.
        max_val (float): Maximum value in the dataset.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    return normalized_data, min_val, max_val


def process_and_split_data(filename: str, noise_std=1.0):
    """
    Load dataset from HDF5 file, normalize features, and split into train/test sets.

    Args:
        filename (str): Path to the HDF5 dataset file.

    Returns:
        train_data (dict): Dictionary of training data per K.
        test_data (dict): Dictionary of test data per K.
    """
    with h5py.File(filename, "r") as hdf_file:
        ap_positions = np.array(hdf_file["AP_positions"])
        ap_positions, ap_min, ap_max = manual_normalize_feature(ap_positions)

        train_data = {}
        val_data = {}

        for K in hdf_file["data"]:
            # Load user data for this K
            user_data = {key: np.array(hdf_file[f"data/{K}/{key}"])
                         for key in hdf_file[f"data/{K}"]}
            num_samples = user_data["user_positions"].shape[0]

            # --- Add Gaussian noise to ALL user positions ---
            noisy_user_positions = user_data["user_positions"] + np.random.normal(
                loc=0.0, scale=noise_std, size=user_data["user_positions"].shape
            )

            # Normalize noisy user positions
            user_positions, user_min, user_max = manual_normalize_feature(noisy_user_positions)

            # Normalize UL and DL powers
            ul_powers, ul_min, ul_max = manual_normalize_global(user_data["optimal_UL_powers"])
            dl_powers, dl_min, dl_max = manual_normalize_global(user_data["optimal_DL_powers"])
            y = np.hstack([ul_powers, dl_powers])  # shape: (samples, 2 * num_users)

            # Broadcast AP positions to match sample count
            ap_positions_broadcasted = np.tile(ap_positions, (num_samples, 1, 1))  # (samples, num_APs, 2)

            # Train/test split
            up_train, up_val, ap_train, ap_val, y_train, y_val, indices_train, indices_val = train_test_split(
                user_positions, ap_positions_broadcasted, y, np.arange(num_samples),
                test_size=0.1, random_state=42
            )

            train_data[K] = {
                "user_positions": up_train,
                "ap_positions": ap_train,
                "y_train": y_train,
            }

            val_data[K] = {
                "user_positions": up_val,
                "ap_positions": ap_val,
                "y_test": y_val,
                "ul_min": ul_min,
                "ul_max": ul_max,
                "dl_min": dl_min,
                "dl_max": dl_max,
            }

    # Sort by K (convert keys to int for ordering)
    train_data = OrderedDict(sorted(train_data.items(), key=lambda item: int(item[0])))
    val_data = OrderedDict(sorted(val_data.items(), key=lambda item: int(item[0])))

    return train_data, val_data


if __name__ == "__main__":
    filename = "cell_free_mMIMO_data.hdf5"
    train_data, val_data = process_and_split_data(filename)

