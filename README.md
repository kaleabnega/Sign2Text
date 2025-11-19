# Arabic Sign Language Word-Level Detection (KArSL)

This notebook-driven project trains a temporal CNN to recognize Arabic Sign Language (ArSL) words using the **KArSL** dataset:

> Sidig, A. A. I., Luqman, H., Mahmoud, S., & Mohandes, M. (2021). KArSL: Arabic Sign Language Database. _ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)_, 20(1), Article 14. https://doi.org/10.1145/3423420

KArSL provides frame-level recordings for dozens of ArSL letters and words. Each word has multiple instances, and every instance is a folder of sequential RGB frames. The notebook `arsl-word-level-detection.ipynb` converts those frames into MediaPipe hand landmarks, caches them as temporal tensors, trains a lightweight temporal CNN, and serves both batch and live webcam inference.

## Project Overview

1. **Frame-to-Landmark Conversion**

   - Load every frame in an instance folder, sort them, and uniformly resample to a fixed `sequence_length` (default 32 frames).
   - Run **MediaPipe Hands** on each frame, normalize the 21×3 landmarks for left and right hands, and flatten them into a per-frame feature vector.
   - Stack the frame features to get a `(T × feature_dim)` tensor for each instance.
   - Save `train_sequences.npy`, `train_labels.npy`, `val_sequences.npy`, `val_labels.npy`, and `label_map.json` in `arsl-word-level-detection/cache/`.

2. **Temporal CNN Classifier**

   - Uses stacked 1D convolutions (TemporalConvNet) with BatchNorm, ReLU, and dropout.
   - Global average pooling over time feeds a linear classifier for the final prediction.
   - Training loop tracks both loss and accuracy per epoch; the notebook now exposes `plot_training_history(history)` for visualizing convergence.

3. **Inference Modes**
   - `predict_instance(Path(...), model, label_map)` classifies any folder of frames.
   - `run_live_webcam_inference()` captures webcam frames, maintains a rolling buffer, and streams real-time predictions (press `q` to exit).

## Files

```
arsl-word-level-detection/
├── arsl-word-level-detection.ipynb   # Main notebook with preprocessing, training, evaluation, live inference
├── organize-dataset.py               # Helper to organize KArSL videos/frames if needed
├── cache/                            # Generated `.npy` caches + label map + trained weights
└── README.md                         # This document
```

## Setup

1. **Environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install mediapipe opencv-python torch torchvision torchaudio pandas scikit-learn matplotlib numpy
   ```

2. **Dataset Placement**
   - Download KArSL (train/val splits) and place it under a root directory with the structure:
     ```
     ARSL_dataset/
       ├── train/
       │   ├── WORD_A/
       │   │   ├── instance_001/
       │   │   │   ├── frame_0001.jpg
       │   │   │   └── ...
       │   │   └── instance_040/
       │   └── WORD_B/
       └── val/
           └── ...
     ```
   - Update `ARSLConfig.dataset_root` in the notebook to point to this directory.

## Workflow

1. **Configure**  
   Adjust `ARSLConfig` parameters (dataset path, sequence length, training hyperparameters) in the first code cells.

2. **Cache Landmarks**  
   Run `generate_all_caches(overwrite=True)` to convert every instance into landmark sequences. The caches are stored in `arsl-word-level-detection/cache/`.

3. **Train**  
   Execute `model, history = full_training_pipeline()` to:

   - Regenerate caches if requested
   - Build PyTorch DataLoaders for train/val splits
   - Train the temporal CNN while logging metrics
   - Save weights to `cache/arsl_tcn.pt`

4. **Visualize**  
   Call `plot_training_history(history)` to display training vs. validation loss curves.

5. **Evaluate and Infer**

   - Use `predict_instance(Path('/path/to/train/WORD/instance_x'), model, label_map)` to classify any cached instance.
   - Run `run_live_webcam_inference()` for live predictions on a local machine.

6. **Deploy/Extend**  
   Export the saved weights (`arsl_tcn.pt`) for deployment, or extend the model with LSTMs/Transformers if longer sequences or multimodal inputs are needed.

## Future Improvements

- **Augmentation**: KArSL includes variations across signers, but you can add landmark jitter or random frame dropping to improve generalization.
- **Sequence Length**: Increase `sequence_length` if you want finer granularity; regenerate caches afterward.
- **Holistic Landmarks**: For whole-body cues (some ArSL words involve shoulders/face), switch from `mp.solutions.hands` to `mp.solutions.holistic` and expand the feature vector.

## Citation

If you use this work or the KArSL dataset, please cite:

```
@article{sidig2021karsl,
  title={KArSL: Arabic Sign Language Database},
  author={Sidig, Alamin Abdelhakam Idris and Luqman, Hafizah and Mahmoud, Samhaa R. and Mohandes, Muhammad},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  volume={20},
  number={1},
  pages={1--19},
  year={2021},
  doi={10.1145/3423420}
}
```

Feel free to adapt the notebook for new ArSL words, additional sensors, or deployment targets.
