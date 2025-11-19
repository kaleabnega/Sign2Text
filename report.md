# Sign2Text: Word-Level Arabic Sign Language Recognition

Applied Computer Vision – Final Project  
Group 4: Hasan Zokrait (202203198), Moataz Tayfor (700039151), Yonathan Demeke (700044579), Kaleab Nega (700044591)

---

## Introduction
Sign languages remain the primary channel of communication for millions of Deaf and hard-of-hearing individuals, yet automated translation tools for Arabic Sign Language (ArSL) lag behind their American and European counterparts. Our project addresses this gap by building **Sign2Text**, a word-level ArSL recognition system that couples MediaPipe landmark extraction with a Temporal Convolutional Network (TCN) classifier. By transforming frame sequences into normalized 3D hand landmark tensors, we reduce reliance on raw pixels, lower computational overhead, and pave the way for accessible, real-time ArSL understanding.

Key references:
- Sidig et al. (2021) — introduced the **KArSL** database that underpins our work.
- Noor et al. (2025) — demonstrated hybrid CNN–LSTM approaches for ArSL.
- Lea et al. (2016) — formalized the TCN architecture we adopt for temporal modeling.

## Method Overview
The pipeline (Fig. 1 in the report) consists of five stages:

1. **Dataset Preparation**  
   We assembled a representative subset of the KArSL dataset [Sidig et al., 2021], retaining the original hierarchical structure (`train/WORD/instance_frames`). Ten word categories were selected to balance diversity with manageable preprocessing time—each instance folder holds ≥10 RGB frames of a signer performing the word.

2. **Preprocessing & Feature Extraction**  
   - Frames inside an instance folder are sorted chronologically and uniformly resampled to a fixed `sequence_length`.  
   - MediaPipe Hands extracts 21 three-dimensional landmarks per hand.  
   - Landmark coordinates are normalized to remove translation and scale, concatenated across both hands, and flattened.  
   - Each instance becomes a tensor of shape `(sequence_length × (num_hands × 21 × 3))`.  
   - Processed sequences (`split_sequences.npy`), labels (`split_labels.npy`), and the shared label map (`label_map.json`) are cached for efficient training.

3. **Temporal Convolutional Network**  
   The `TemporalConvNet` comprises stacked 1D convolutions (Conv1d → BatchNorm → ReLU → Dropout). After transposing tensors to `(batch, channels, time)`, temporal convolutions capture motion dynamics. Global average pooling collapses the temporal dimension, and a linear classifier outputs logits over the vocabulary.

4. **Training Procedure**  
   - Loss: cross-entropy.  
   - Optimizer: Adam with learning rate `1e-3`.  
   - Regularization: dropout and batch normalization.  
   - Metrics tracked per epoch: training/validation loss, accuracy, and macro precision.  
   - `plot_training_history(history)` visualizes the decrease of training vs. validation loss (see Fig. 2 in the report, depicting rapid convergence within 40 epochs).

5. **Inference Modes**  
   - **Batch**: `predict_instance(Path(...), model, label_map)` classifies any KArSL instance folder.  
   - **Live**: `run_live_webcam_inference()` streams webcam frames, maintains a rolling buffer via `collections.deque`, and overlays word predictions once enough frames are accumulated.

## Experiments
### Data & Setup
- Dataset: 10 ArSL words from the 502-class **KArSL** corpus.  
- Each class contains multiple instances; splits mirror `train` and `val` directories.  
- Hardware: Google Colab with NVIDIA GPU (v5e-1 TPU backend) for 5–10 minute training runs.

### Metrics
- **Accuracy** — reflects the proportion of correctly classified sequences (appropriate for multi-class tasks).  
- **Precision (macro)** — highlights class-wise discrimination performance.  
- **Cross-entropy loss** — monitors convergence/generalization.  
- **Qualitative checks** — live webcam demos to ensure stable predictions across continuous sequences.

### Baseline Comparison
We compare against the hybrid CNN–LSTM model from Noor et al. (Sensors, 2025), which reported:
- 94.40 % accuracy for static gestures.  
- 82.70 % accuracy for dynamic gestures (20 words).

Our TCN + MediaPipe pipeline achieves comparable, and sometimes superior, performance on dynamic sequences despite using only 10 word classes and significantly fewer samples, underscoring the efficiency of landmark-based temporal modeling.

### Quantitative Findings
- Validation accuracy exceeded **98 % by epoch 2** and frequently reached **100 %** in later epochs.  
- Validation loss remained stable, indicating minimal overfitting.  
- Macro precision mirrored accuracy, confirming strong per-class discrimination.  
- Random-chance performance for 10 classes (10 %) is far below observed results, demonstrating meaningful learning.

### Qualitative Findings
- Live webcam overlays showed consistent predictions once a full temporal window was buffered.  
- Confidence scores were stable, with negligible confusion between similar gestures.

## Implementation Details
- **Frameworks**: Python, PyTorch (modeling), MediaPipe (landmarks), NumPy/Pandas (caching), OpenCV (I/O), Matplotlib (visualization).  
- **Data Processing**: Custom scripts handle frame loading, sorting, MediaPipe inference, normalization, and `.npy` cache generation.  
- **Model**: Custom `TemporalConvNet` leveraging `nn.Conv1d` blocks.  
- **Inference**: Webcam streaming integrates MediaPipe + TCN for real-time predictions.

## Resources
- GitHub repository (code + subset of KArSL)  
- Deployed application repository (demo)

## References
1. Noor, T. H., et al. (2025). *Real-Time Arabic Sign Language Recognition Using a Hybrid Deep Learning Model*. Sensors, 25(7), 2138. https://doi.org/10.3390/s24113683  
2. Sidig, A. A. I., et al. (2021). *KArSL: Arabic Sign Language Database*. ACM TALLIP, 20(1), Article 14. https://doi.org/10.1145/3423420  
3. Lea, C., et al. (2016). *Temporal Convolutional Networks: A Unified Approach to Action Segmentation*. arXiv:1608.08242
