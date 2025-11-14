# %% [markdown]
"""
## ASL Letter Classifier From MediaPipe Landmarks

This notebook-style script walks through collecting hand landmarks with MediaPipe, organizing them into a dataset, training a PyTorch multilayer perceptron to classify ASL alphabet letters, and running the trained model on live webcam frames. Each cell is annotated to clarify the pipeline end-to-end.
"""

# %%
import json  # Enables saving metadata such as label maps in JSON format
import os  # Offers filesystem helpers for creating folders and joining paths
import time  # Supplies timestamps for throttling capture loops
from dataclasses import dataclass  # Provides structured containers for configuration
from pathlib import Path  # Makes path manipulations easier and cross-platform
from typing import Dict, List, Optional, Tuple  # Adds helpful type hints for clarity

import cv2  # Handles webcam capture and basic drawing on frames
import mediapipe as mp  # Provides pretrained landmark detectors for hands
import numpy as np  # Powers vectorized math on landmark tensors
import pandas as pd  # Simplifies CSV I/O for the collected dataset
import torch  # Hosts tensors and neural network operations
from sklearn.model_selection import train_test_split  # Splits data consistently
from torch import nn  # Supplies building blocks for the classifier
from torch.utils.data import DataLoader, Dataset  # Creates iterable datasets

# %%
@dataclass
class TrainingConfig:
    """Holds all tunable knobs for data paths and model hyperparameters."""

    data_dir: Path = Path("data/asl_landmarks")  # Folder where CSVs and artifacts live
    landmark_csv: str = "asl_letters.csv"  # Relative CSV file storing flattened landmarks
    label_map_json: str = "label_map.json"  # JSON file that records label-to-index mapping
    raw_dataset_dir: Path = Path("asl_dataset")  # Location of the RGB dataset with class folders
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")  # Supported RGB file types
    num_keypoints: int = 21  # Number of landmarks MediaPipe Hands outputs for one hand
    landmark_dims: int = 3  # Each keypoint has x, y, z coordinates
    test_size: float = 0.2  # Validation split ratio for train/validation separation
    random_state: int = 42  # Random seed for deterministic shuffling
    hidden_sizes: Tuple[int, int] = (256, 128)  # Hidden layer widths for the MLP
    dropout: float = 0.3  # Dropout probability for regularization
    batch_size: int = 64  # Number of samples per training batch
    num_epochs: int = 30  # How many full passes through the training set
    learning_rate: float = 1e-3  # Adam optimizer initial learning rate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-pick accelerator


# %%
config = TrainingConfig()  # Instantiate the configuration with defaults
config.data_dir.mkdir(parents=True, exist_ok=True)  # Ensure the data directory exists
print(f"Using device: {config.device}")  # Log the compute target for transparency


# %%
mp_hands = mp.solutions.hands  # Shortcut handle for the MediaPipe Hands solution
mp_drawing = mp.solutions.drawing_utils  # Utility for visualizing landmarks on frames


# %%
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate and scale landmarks so the model is invariant to absolute position."""

    centered = landmarks - landmarks.mean(axis=0, keepdims=True)  # Center around origin
    norm = np.linalg.norm(centered) + 1e-6  # Compute magnitude with epsilon to avoid zero
    normalized = centered / norm  # Scale to unit length for consistency
    return normalized  # Return normalized coordinates


# %%
def is_valid_label_folder(folder_name: str) -> bool:
    """Check whether a folder name represents a single alphanumeric class label."""

    cleaned = folder_name.strip().lower()  # Normalize whitespace and case
    return len(cleaned) == 1 and cleaned.isalnum()  # Accept single letters or digits only


# %%
def extract_landmarks_from_image(
    image_path: Path,
    hands: mp.solutions.hands.Hands,
) -> Optional[np.ndarray]:
    """Run MediaPipe on a static image and return normalized landmarks if a hand is detected."""

    image = cv2.imread(str(image_path))  # Load image from disk
    if image is None:  # Validate successful load
        print(f"Warning: failed to read {image_path}")  # Inform about the issue
        return None  # Skip this sample
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = hands.process(image_rgb)  # Execute landmark detection
    if not results.multi_hand_landmarks:  # Check if any hand was found
        return None  # Return None to signal a skipped frame
    coords = np.array(  # Build numpy array of landmark coordinates
        [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark],
        dtype=np.float32,
    )
    coords = normalize_landmarks(coords)  # Apply normalization for invariance
    return coords  # Return prepared coordinates


# %%
def generate_landmark_dataset_from_rgb(overwrite: bool = False) -> None:
    """Convert an existing folder-structured RGB dataset into a landmark CSV via MediaPipe."""

    dataset_root = config.raw_dataset_dir  # Resolve dataset root path
    if not dataset_root.exists():  # Ensure the dataset folder is present
        raise FileNotFoundError(f"RGB dataset folder {dataset_root} does not exist.")  # Guard clause

    csv_path = config.data_dir / config.landmark_csv  # Target CSV path
    label_map_path = config.data_dir / config.label_map_json  # Target label map path
    if csv_path.exists() and not overwrite:  # Prevent accidental overwrite
        raise FileExistsError(
            f"{csv_path} already exists. Pass overwrite=True to rebuild it from scratch."
        )

    label_dirs = sorted(  # Collect valid label directories
        [
            d
            for d in dataset_root.iterdir()
            if d.is_dir() and is_valid_label_folder(d.name)
        ],
        key=lambda d: d.name.lower(),
    )
    if not label_dirs:  # Guard against empty datasets
        raise ValueError(f"No label folders found inside {dataset_root}.")

    label_names = [d.name.strip().lower() for d in label_dirs]  # Normalize label names
    label_map = {label: idx for idx, label in enumerate(label_names)}  # Build deterministic map
    label_map_path.write_text(json.dumps(label_map, indent=2))  # Persist mapping for later use

    records: List[Dict[str, float]] = []  # Accumulate flattened landmark rows
    stats = {"total_images": 0, "samples_saved": 0, "skipped_no_hand": 0}  # Track progress stats

    with mp_hands.Hands(  # Initialize MediaPipe in static image mode
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        for label_dir in label_dirs:  # Iterate over every label folder
            label = label_dir.name.strip().lower()  # Normalize current label
            label_index = label_map[label]  # Look up integer index
            image_files = sorted(label_dir.glob("*"))  # List files in the folder
            for image_path in image_files:  # Loop through each file
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in config.image_extensions
                ):  # Filter unsupported entries
                    continue  # Skip non-image files early
                stats["total_images"] += 1  # Increment total counter
                coords = extract_landmarks_from_image(image_path, hands)  # Run MediaPipe
                if coords is None:  # Handle missing detections
                    stats["skipped_no_hand"] += 1  # Record skip reason
                    continue  # Move on to next image
                flat = coords.flatten()  # Flatten landmarks to 1D vector
                sample_row = {f"kp_{i}": float(value) for i, value in enumerate(flat)}  # Build row dict
                sample_row["label"] = label  # Store original label text
                sample_row["label_index"] = label_index  # Store numerical label
                records.append(sample_row)  # Append row to memory
                stats["samples_saved"] += 1  # Update saved counter

    if not records:  # Ensure we gathered usable samples
        raise RuntimeError("No landmarks were extracted. Check dataset quality or MediaPipe settings.")

    df = pd.DataFrame(records)  # Convert collected rows into a DataFrame
    df.to_csv(csv_path, index=False)  # Persist the dataset to CSV once
    print(  # Summarize extraction results
        f"Landmark CSV created at {csv_path} | "
        f"processed {stats['total_images']} images, "
        f"saved {stats['samples_saved']} samples, "
        f"skipped {stats['skipped_no_hand']} without detectable hands."
    )


# %%
def record_letter_samples(
    letter_label: str,
    num_samples: int,
    capture_fps: int = 5,
    camera_index: int = 0,
    mirror_view: bool = True,
) -> None:
    """Capture MediaPipe landmarks for a single ASL letter and append them to the CSV."""

    csv_path = config.data_dir / config.landmark_csv  # Resolve dataset CSV path
    label_map_path = config.data_dir / config.label_map_json  # Resolve label map path
    label_map = {}  # Initialize label map dictionary
    if label_map_path.exists():  # Check if label map already exists
        label_map = json.loads(label_map_path.read_text())  # Load existing map from disk
    if letter_label not in label_map:  # Add new label if unseen
        label_map[letter_label] = len(label_map)  # Assign next available index
        label_map_path.write_text(json.dumps(label_map, indent=2))  # Persist mapping

    cap = cv2.VideoCapture(camera_index)  # Initialize webcam capture
    if not cap.isOpened():  # Guard clause if webcam fails to open
        raise RuntimeError("Could not access webcam")  # Raise descriptive error

    samples_collected = 0  # Counter for successful samples
    frame_delay = 1.0 / capture_fps  # Compute delay between samples for target FPS
    last_capture_time = 0.0  # Track time of the previous capture

    with mp_hands.Hands(  # Initialize MediaPipe Hands context manager
        static_image_mode=False,  # Use video mode for faster inference
        max_num_hands=1,  # Focus on the dominant hand per frame
        min_detection_confidence=0.7,  # Detection threshold
        min_tracking_confidence=0.7,  # Tracking threshold
    ) as hands:
        while samples_collected < num_samples:  # Loop until target count reached
            ret, frame = cap.read()  # Grab the next frame from the webcam
            if not ret:  # Handle camera read failure
                print("Frame grab failed, retrying...")  # Provide feedback
                continue  # Skip to next loop iteration

            if mirror_view:  # Optionally mirror the frame for natural feel
                frame = cv2.flip(frame, 1)  # Flip along the vertical axis

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
            results = hands.process(rgb_frame)  # Run landmark detection

            if results.multi_hand_landmarks:  # Proceed only when a hand is detected
                hand_landmarks = results.multi_hand_landmarks[0]  # Use first detected hand
                mp_drawing.draw_landmarks(  # Overlay landmarks for visual feedback
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                now = time.time()  # Capture current timestamp
                if now - last_capture_time >= frame_delay:  # Enforce sampling interval
                    last_capture_time = now  # Update capture timestamp
                    coords = np.array(  # Convert landmarks to numpy array
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    )
                    coords = normalize_landmarks(coords)  # Normalize for invariance
                    flat = coords.flatten()  # Flatten to 1D vector
                    sample_row = {  # Build dictionary for DataFrame append
                        f"kp_{i}": value for i, value in enumerate(flat)
                    }
                    sample_row["label"] = letter_label  # Attach label string
                    sample_row["label_index"] = label_map[letter_label]  # Attach label index
                    df_row = pd.DataFrame([sample_row])  # Convert to single-row DataFrame
                    header = not csv_path.exists()  # Determine if header should be written
                    df_row.to_csv(  # Append sample to CSV
                        csv_path,
                        mode="a",
                        header=header,
                        index=False,
                    )
                    samples_collected += 1  # Increment collection counter
                    print(f"Captured {samples_collected}/{num_samples} for {letter_label}")  # Log progress

            cv2.putText(  # Overlay instructions on the frame
                frame,
                f"Recording letter: {letter_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("ASL Capture", frame)  # Display the live frame
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Provide quit shortcut
                break  # Exit capture loop early

    cap.release()  # Release webcam resource
    cv2.destroyAllWindows()  # Close display windows


# %%
def load_dataset() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load the CSV dataset and associated label map from disk."""

    csv_path = config.data_dir / config.landmark_csv  # Resolve CSV path
    label_map_path = config.data_dir / config.label_map_json  # Resolve label map path
    if not csv_path.exists():  # Validate dataset presence
        raise FileNotFoundError("Dataset CSV not found. Capture samples first.")  # Guard clause
    if not label_map_path.exists():  # Validate label map presence
        raise FileNotFoundError("Label map JSON missing.")  # Guard clause
    df = pd.read_csv(csv_path)  # Read dataset into DataFrame
    label_map = json.loads(label_map_path.read_text())  # Load label indices
    return df, label_map  # Return both structures


# %%
class ASLLandmarkDataset(Dataset):
    """PyTorch Dataset that converts landmark rows into tensors."""

    def __init__(self, frame: pd.DataFrame):
        self.features = frame.filter(like="kp_").values.astype(np.float32)  # Extract features
        self.labels = frame["label_index"].values.astype(np.int64)  # Extract integer labels

    def __len__(self) -> int:
        return len(self.labels)  # Return dataset size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(self.features[idx])  # Convert feature row to tensor
        label = torch.tensor(self.labels[idx])  # Convert label to tensor
        return feature, label  # Provide tuple for DataLoader


# %%
class LandmarkMLP(nn.Module):
    """Simple multilayer perceptron for ASL letter classification."""

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int], num_classes: int, dropout: float):
        super().__init__()  # Initialize parent class
        self.model = nn.Sequential(  # Stack layers sequentially
            nn.Linear(input_dim, hidden_sizes[0]),  # First dense layer
            nn.BatchNorm1d(hidden_sizes[0]),  # Normalize activations
            nn.ReLU(),  # Non-linearity
            nn.Dropout(dropout),  # Regularization
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),  # Second dense layer
            nn.BatchNorm1d(hidden_sizes[1]),  # Normalize activations
            nn.ReLU(),  # Non-linearity
            nn.Dropout(dropout),  # Regularization
            nn.Linear(hidden_sizes[1], num_classes),  # Output logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Forward pass through sequential stack


# %%
def prepare_dataloaders() -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create train and validation DataLoaders from the CSV dataset."""

    df, label_map = load_dataset()  # Load raw dataset and label map
    train_df, val_df = train_test_split(  # Split DataFrame into train and val
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label_index"],
    )
    train_dataset = ASLLandmarkDataset(train_df.reset_index(drop=True))  # Build train dataset
    val_dataset = ASLLandmarkDataset(val_df.reset_index(drop=True))  # Build val dataset
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)  # Train loader
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)  # Val loader
    return train_loader, val_loader, label_map  # Return loaders and label mapping


# %%
def train_model(train_loader: DataLoader, val_loader: DataLoader, num_classes: int) -> nn.Module:
    """Train the LandmarkMLP and return the best-performing model."""

    input_dim = config.num_keypoints * config.landmark_dims  # Compute flattened size
    model = LandmarkMLP(input_dim, config.hidden_sizes, num_classes, config.dropout).to(config.device)  # Init model
    criterion = nn.CrossEntropyLoss()  # Classification loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # Optimizer

    best_val_accuracy = 0.0  # Track best validation accuracy
    best_state = None  # Placeholder for best model weights

    for epoch in range(1, config.num_epochs + 1):  # Loop through epochs
        model.train()  # Set model to training mode
        running_loss = 0.0  # Accumulate training loss
        running_correct = 0  # Accumulate correct predictions
        running_total = 0  # Accumulate sample count

        for features, labels in train_loader:  # Iterate over training batches
            features = features.to(config.device)  # Move features to device
            labels = labels.to(config.device)  # Move labels to device
            optimizer.zero_grad()  # Reset gradients
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            running_loss += loss.item() * features.size(0)  # Accumulate scaled loss
            preds = outputs.argmax(dim=1)  # Get predicted classes
            running_correct += (preds == labels).sum().item()  # Count correct predictions
            running_total += labels.size(0)  # Update total count

        train_loss = running_loss / running_total  # Average training loss
        train_acc = running_correct / running_total  # Training accuracy

        val_loss, val_acc = evaluate(model, val_loader, criterion)  # Evaluate on validation set

        print(  # Log metrics for the epoch
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_accuracy:  # Check for improvement
            best_val_accuracy = val_acc  # Update best accuracy
            best_state = model.state_dict()  # Snapshot model weights

    if best_state is not None:  # Ensure at least one improvement occurred
        model.load_state_dict(best_state)  # Restore best weights
    return model  # Return trained model


# %%
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Compute average loss and accuracy for a given DataLoader."""

    model.eval()  # Set model to evaluation mode
    total_loss = 0.0  # Accumulate loss
    total_correct = 0  # Accumulate correct predictions
    total_samples = 0  # Accumulate number of samples

    with torch.no_grad():  # Disable gradient computation
        for features, labels in loader:  # Iterate over loader
            features = features.to(config.device)  # Move to device
            labels = labels.to(config.device)  # Move to device
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            total_loss += loss.item() * features.size(0)  # Accumulate scaled loss
            preds = outputs.argmax(dim=1)  # Predicted classes
            total_correct += (preds == labels).sum().item()  # Count corrects
            total_samples += labels.size(0)  # Update sample count

    avg_loss = total_loss / total_samples  # Average loss
    avg_acc = total_correct / total_samples  # Average accuracy
    return avg_loss, avg_acc  # Return both metrics


# %%
def save_model(model: nn.Module) -> Path:
    """Persist the trained model weights to disk."""

    model_path = config.data_dir / "asl_letter_mlp.pt"  # Define weights path
    torch.save(model.state_dict(), model_path)  # Save state dict
    print(f"Saved model to {model_path}")  # Confirm save location
    return model_path  # Return path for reference


# %%
def load_model(label_map: Dict[str, int]) -> nn.Module:
    """Reload a trained model for inference."""

    model_path = config.data_dir / "asl_letter_mlp.pt"  # Define weights path
    if not model_path.exists():  # Ensure weights file exists
        raise FileNotFoundError("Trained model not found. Train the model first.")  # Guard clause
    num_classes = len(label_map)  # Determine number of classes
    input_dim = config.num_keypoints * config.landmark_dims  # Determine input dimension
    model = LandmarkMLP(input_dim, config.hidden_sizes, num_classes, config.dropout).to(config.device)  # Instantiate
    model.load_state_dict(torch.load(model_path, map_location=config.device))  # Load weights
    model.eval()  # Set to eval mode
    return model  # Return model ready for inference


# %%
def run_live_inference() -> None:
    """Stream webcam frames, extract landmarks, and display letter predictions."""

    _, label_map = load_dataset()  # Load label map to rebuild indices
    inv_label_map = {idx: label for label, idx in label_map.items()}  # Reverse mapping
    model = load_model(label_map)  # Load trained model

    cap = cv2.VideoCapture(0)  # Initialize webcam
    if not cap.isOpened():  # Check if webcam accessible
        raise RuntimeError("Could not access webcam")  # Guard clause

    with mp_hands.Hands(  # Prepare MediaPipe Hands
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:  # Continuous stream loop
            ret, frame = cap.read()  # Capture frame
            if not ret:  # Validate capture
                print("Frame grab failed, exiting stream.")  # Log issue
                break  # Exit loop

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            results = hands.process(rgb_frame)  # Extract landmarks

            if results.multi_hand_landmarks:  # If hand detected
                hand_landmarks = results.multi_hand_landmarks[0]  # Take first hand
                mp_drawing.draw_landmarks(  # Draw landmarks for visualization
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])  # To numpy
                coords = normalize_landmarks(coords)  # Normalize
                features = torch.from_numpy(coords.flatten().astype(np.float32)).unsqueeze(0)  # To tensor batch
                features = features.to(config.device)  # Move to device
                with torch.no_grad():  # Disable gradients
                    logits = model(features)  # Forward pass
                    probs = torch.softmax(logits, dim=1)  # Convert to probabilities
                    pred_idx = int(torch.argmax(probs, dim=1).item())  # Argmax label
                    pred_letter = inv_label_map.get(pred_idx, "?")  # Map to letter
                    confidence = float(torch.max(probs).item())  # Confidence score
                cv2.putText(  # Overlay prediction on frame
                    frame,
                    f"{pred_letter} ({confidence:.2f})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:  # If no hand detected
                cv2.putText(  # Prompt user to show hand
                    frame,
                    "Show hand to detect letter",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("ASL Letter Inference", frame)  # Display annotated frame
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Provide quit key
                break  # Exit loop

    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close windows


# %%
def full_training_pipeline() -> nn.Module:
    """Convenience helper to prepare data, train, evaluate, and save the model."""

    train_loader, val_loader, label_map = prepare_dataloaders()  # Prepare data
    num_classes = len(label_map)  # Determine number of classes
    model = train_model(train_loader, val_loader, num_classes)  # Train model
    save_model(model)  # Persist weights
    return model  # Return trained model for immediate use


# %%
if __name__ == "__main__":
    print(
        "This file is meant to be run in sections like a notebook.\n"
        "To start, call record_letter_samples(...) letter by letter, then run full_training_pipeline(),\n"
        "and finally run_live_inference() once a model checkpoint exists."
    )
