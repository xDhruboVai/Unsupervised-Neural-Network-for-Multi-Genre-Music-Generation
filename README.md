**Generated Midis:** [Google Drive Folder](https://drive.google.com/drive/folders/1i__ie7kPIMU3N3Ew-3vY23F4RXuvDX6V?usp=sharing)
**Report:** [Google Drive Folder](https://drive.google.com/file/d/1J2Gs-hl2Os3AsI-J-n3beCLFst7i8ruA/view?usp=sharing)

# CSE425 Project: Generative Piano Music with MAESTRO

## Overview
This repository implements a full symbolic music generation pipeline using the MAESTRO dataset. The project covers data extraction, exploratory analysis, preprocessing, tokenization, and three model families:

- Task 1: LSTM Autoencoder (piano-roll domain)
- Task 2: LSTM Variational Autoencoder with KL annealing (piano-roll domain)
- Task 3: Genre-conditioned Transformer decoder on REMI token sequences

All training and evaluation workflows are notebook-driven, and the repository includes saved artifacts (models, generated MIDI, metrics, and plots) for each task.

## What This Project Does
At a high level, the project supports two complementary representations of music:

1. Piano-roll representation for Tasks 1 and 2
- Source MIDI is converted into binary windows of shape (128 timesteps, 88 keys)
- Frame rate fs=16, so each 128-step window is about 8 seconds
- Low-activity windows are filtered out with a 2% activity threshold

2. REMI token representation for Task 3
- MIDI is tokenized with MidiTok REMI
- Fixed token windows of length 512 with stride 256
- Genre labels are attached per sequence (3-class composer-era grouping)

The repository then trains generative models and exports MIDI outputs for qualitative listening.

## Repository Layout

- data/
  - raw_data/: MAESTRO MIDI files and metadata
  - train-val-test/: piano-roll windows for Tasks 1/2
  - tokenized/: REMI token arrays, genre arrays, tokenizer config
- notebooks/
  - 0.0_Extract.ipynb: dataset extraction
  - 0.1_EDA.ipynb: exploratory analysis
  - 0.2_preprocessing.ipynb: piano-roll window extraction
  - 0.3_tokenization.ipynb: REMI tokenization + genre labeling
  - 1_LSTM_AE.ipynb: Task 1 model training/generation
  - 2_VAE.ipynb: Task 2 model training/generation/interpolation/evaluation
  - 3_transformer.ipynb: Task 3 model training and generation
  - 3_transformer_pc.ipynb: local inference/generation helper
- output/
  - EDA/: dataset-level plots
  - tokenization/: token/genre distribution plots
  - task1/: AE checkpoint, loss curve, generated MIDI
  - task2/: VAE checkpoints, history, metrics, generated/interpolated MIDI
  - task3/: Transformer checkpoint/history, generated MIDI
- requirements.txt
- sustain_midi_notes.py: MIDI timing post-processing utility
- example_sustain.py: usage examples for MIDI timing utilities

## Data and Preprocessing Summary

### MAESTRO EDA highlights (from saved plots)
- Piece durations are highly variable and right-skewed.
  - Min: 115.5 s
  - Max: 2398.9 s
  - Mean: 645.3 s
  - Median: 548.4 s
- Notes per file are also right-skewed.
  - Min: 875
  - Max: 19689
  - Mean: 6392
  - Median: 5255
  - Mean note density: 10.1 notes/sec
- Velocity distribution is centered in the mid-range.
  - Min: 6
  - Max: 125
  - Mean: 64.6
  - Median: 66.0
  - Std: 19.0
- Pitch distribution is concentrated in middle registers, with tapering tails at low and high extremes.

### Piano-roll dataset produced (Task 1/2 input)
- Train: (62689, 128, 88)
- Validation: (7876, 128, 88)
- Test: (7792, 128, 88)
- Data type: float32 binary activation windows

### Sparsity profile (from saved plots)
- Mean sparsity: 94.46%
- Mean activity: 5.54%
- Filtering threshold: windows kept at >=2% activity

This is a sparse-sequence generation setting; loss design and decoding constraints matter significantly.

## Tokenization and Genre Conditioning (Task 3 input)

### Tokenization config
- Tokenizer: REMI (MidiTok)
- Vocab size: 284
- PAD/BOS/EOS IDs: 0 / 1 / 2
- Max sequence length: 512
- Stride: 256
- Velocity bins: 32

### Token datasets produced
- tokens_train.npy: (80449, 512)
- tokens_validation.npy: (9103, 512)
- tokens_test.npy: (10487, 512)

### Genre labels
Three composer-era buckets are used:
- 0: Baroque/Classical
- 1: Romantic
- 2: 20th Century/Modern

Train genre distribution from saved EDA:
- Baroque/Classical: 9922
- Romantic: 63865
- 20th Century/Modern: 6662

This is strongly imbalanced toward Romantic-era material.

## Model Implementations

## Task 1: LSTM Autoencoder
- Encoder/decoder hidden size: 256
- Latent size: 64
- 2-layer LSTM encoder and decoder
- Loss: Focal loss variant for sparse binary piano-roll reconstruction
- Training: 60 epochs, Adam, ReduceLROnPlateau, gradient clipping

Saved outputs:
- output/task1/best_model.pth
- output/task1/loss_curve.png
- output/task1/generated/sample_01.mid ... sample_05.mid

Observation from loss curve:
- Train loss decreases steadily
- Validation loss improves early and then plateaus, with mild train/val gap by the end
- Indicates stable training, but not enough evidence in this repository for strong generative quality claims on Task 1 alone

## Task 2: LSTM VAE with KL annealing
- Same base LSTM backbone style as Task 1
- Latent variable model with reparameterization
- KL annealing schedule:
  - beta=0 during warmup
  - linear increase toward ~1.0
- Includes random latent sampling and latent interpolation exports

Saved outputs:
- output/task2/best_vae_model.pth
- output/task2/vae_final_model.pth
- output/task2/vae_training_history.json
- output/task2/vae_training_history.png
- output/task2/generated/generated_01.mid ... generated_08.mid
- output/task2/interpolations/interpolation_00_alpha_0.00.mid ... interpolation_07_alpha_1.00.mid
- output/task2/metrics_comparison.csv
- output/task2/metrics_comparison.png
- output/task2/human_survey_results.csv

Key metrics (from saved files):
- Best validation loss: 0.0383 at epoch 6
- Final validation loss: 0.0480
- Test reconstruction loss: 0.0475
- Rhythm diversity:
  - Real sample: 0.6500
  - Generated sample: 0.1481

Interpretation:
- The model reconstructs reasonably but generated rhythm diversity is far below real data, indicating limited rhythmic variety and likely repetitive structure.

Important note on human survey file:
- The notebook explicitly auto-generates participant scores for demonstration.
- Therefore, the mean score (2.84/5) is not a real listening study and should not be treated as a valid human evaluation result.

## Task 3: Genre-conditioned Transformer decoder (REMI)
- Decoder-only Transformer
- Causal self-attention
- Genre embedding added to token representations
- Training length in saved run: 5 epochs

Saved outputs:
- output/task3/best_transformer_model.pth
- output/task3/transformer_training_history.json
- output/task3/generated/task3_sample_01_genre_0.mid ... task3_sample_10_genre_0.mid

Training metrics from saved history:
- Validation loss: 2.3257 -> 1.9955 (5 epochs)
- Validation perplexity: 10.23 -> 7.36 (5 epochs)

Interpretation:
- Training is moving in the right direction.
- The run appears short; further epochs are likely needed before making final claims about generation quality.

## Overall Verdict (Based on Current Artifacts)
The repository demonstrates a complete and technically coherent symbolic music generation workflow, with reproducible data products and trained models across three paradigms.

Current status by task:
- Task 1 (AE): good reconstruction training behavior, but limited objective quality evidence in saved metrics.
- Task 2 (VAE): reconstruction works; diversity gap versus real data is substantial (0.6500 vs 0.1481), suggesting mode restriction/repetition.
- Task 3 (Transformer): promising early perplexity trend, but training appears under-extended (5 epochs), so conclusions should remain provisional.

Most defensible project-level conclusion:
- The pipeline is strong and complete.
- Generative musical quality is not yet fully competitive with real MAESTRO complexity, mainly due to diversity and likely training-depth limitations.

## Reproducing the Workflow

## Environment
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install miditok
```

Note:
- Some notebooks install packages inline; local runs should still ensure miditok and pretty_midi are available.

## Recommended notebook order
1. notebooks/0.0_Extract.ipynb
2. notebooks/0.1_EDA.ipynb
3. notebooks/0.2_preprocessing.ipynb
4. notebooks/0.3_tokenization.ipynb
5. notebooks/1_LSTM_AE.ipynb
6. notebooks/2_VAE.ipynb
7. notebooks/3_transformer.ipynb
8. notebooks/3_transformer_pc.ipynb (optional local generation helper)

## Generated MIDI post-processing utility
The repository includes a utility to adjust note durations and timing after generation.

- sustain_midi_notes.py
- example_sustain.py

It supports:
- fixed sustain duration
- timing shift after first chord
- no-gap onset compression mode

## Limitations and Next Steps

1. Genre imbalance
- Romantic class dominates token windows and can bias generation.

2. Evaluation rigor
- The current human survey artifact is synthetic; replace with real listener data.
- Add stronger symbolic metrics (e.g., pitch-class entropy, tonal tension, n-gram novelty, self-similarity) and compare against real distributions.

3. Transformer training depth
- Extend Task 3 training beyond 5 epochs and monitor held-out perplexity and qualitative outputs.

4. Decoding controls
- Explore controlled sampling (temperature and top-k sweeps, nucleus sampling, repetition penalties).

5. Representation-level improvements
- Test richer REMI settings (tempo/chord/rest tokens where appropriate) and compare against current minimal configuration.

## Attribution
Dataset:
- MAESTRO v3.0.0

This repository is a course project implementation and is intended as a research/educational baseline for symbolic music generation experiments.
