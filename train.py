import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import HybridPowerAllocator  # import your model class


class GammaPowerDataset(Dataset):
    """
    Custom PyTorch Dataset for gamma power allocation.

    Args:
        user_positions (np.ndarray): User positions, shape (samples, num_users, 2).
        ap_positions (np.ndarray): Access point positions, shape (samples, num_APs, 2).
        y (np.ndarray): Target UL/DL powers, shape (samples, 2 * num_users).

    Returns:
        Tuple of (user_positions, ap_positions, y) as torch tensors.
    """
    def __init__(self, user_positions, ap_positions, y):
        self.user_positions = user_positions
        self.ap_positions = ap_positions
        self.y = y

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.user_positions[idx], dtype=torch.float32),
            torch.tensor(self.ap_positions[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


def train_model(
    model: nn.Module,
    loaders_by_K: dict,
    val_loaders_by_K: dict,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    num_epochs: int,
    device: torch.device
):
    """
    Train and validate the HybridPowerAllocator model across multiple user counts (K).

    Args:
        model (nn.Module): The neural network model.
        loaders_by_K (dict): Training DataLoaders keyed by user count K.
        val_loaders_by_K (dict): Validation DataLoaders keyed by user count K.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (CosineAnnealingLR): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run training on.
    """
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_batches = 0

        # === Training across all Ks ===
        for K, loader in loaders_by_K.items():
            for user_positions, ap_positions, y in loader:
                user_positions = user_positions.to(device)
                ap_positions = ap_positions.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(user_positions, ap_positions)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                total_batches += 1

        scheduler.step()
        train_loss /= total_batches

        # === Validation across all Ks ===
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        val_loss_by_K = {}

        with torch.no_grad():
            for K, loader in val_loaders_by_K.items():
                val_loss_K = 0
                val_samples_K = 0

                for user_positions, ap_positions, y in loader:
                    user_positions = user_positions.to(device)
                    ap_positions = ap_positions.to(device)
                    y = y.to(device)

                    outputs = model(user_positions, ap_positions)
                    loss = criterion(outputs, y)

                    val_loss_K += loss.item() * y.size(0)
                    val_samples_K += y.size(0)

                avg_loss_K = val_loss_K / val_samples_K
                val_loss_by_K[K] = avg_loss_K

                total_val_loss += val_loss_K
                total_val_samples += val_samples_K

        avg_val_loss = total_val_loss / total_val_samples

        # === Logging ===
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss (avg across all Ks): {avg_val_loss:.4f}")
        for K in sorted(val_loss_by_K):
            print(f"  Val Loss for {K} users: {val_loss_by_K[K]:.4f}")

        # === Save best model ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print("Saved Best Model!")


if __name__ == "__main__":
    d_model = 64
    ap_embed_dim = 32
    num_heads = 4
    num_encoder_layers = 2
    dropout_rate = 0.1
    learning_rate = 1e-3
    num_epochs = 5
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = HybridPowerAllocator(
        ap_embed_dim=ap_embed_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_encoder_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    from data_processing import process_and_split_data

    # Load and preprocess data
    train_data, val_data = process_and_split_data("cell_free_mMIMO_dataset.hdf5")

    # Create DataLoaders for training
    loaders_by_K = {
        K: DataLoader(
            GammaPowerDataset(data["user_positions"], data["ap_positions"], data["y_train"]),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        for K, data in train_data.items()
    }

    # Create DataLoaders for validation
    val_loaders_by_K = {
        K: DataLoader(
            GammaPowerDataset(data["user_positions"], data["ap_positions"], data["y_test"]),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        for K, data in val_data.items()
    }

    # Run training
    train_model(model, loaders_by_K, val_loaders_by_K, criterion, optimizer, scheduler, num_epochs, device)

