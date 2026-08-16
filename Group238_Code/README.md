# Automatic Question Generation for E-Learning

## NLP Assignment 2 --- Group 238

### Student Details

  Student ID    Student Name                    Contribution %
  ------------- ----------------------------- ----------------
  2025AA05353   SHEETAL PRAKASH BARANWAL                  100%
  2025AA05174   AISHWARYA TEWARI                          100%
  2025AA05002   JADHAV VEDANT BHAGWAN SWATI               100%
  2025AB05063   J RAJESH KUMAR                            100%
  2025AB05120   JAHNAVI KOLLIPARA                         100%

------------------------------------------------------------------------

## 1. Project Overview

This project implements an Automatic Question Generation system for
e-learning study material using a Transformer-based T5 Encoder--Decoder
(Seq2Seq) model.

### Key Features

-   Direct text entry and document parsing (`.txt`, `.pdf`, `.csv`).
-   Target answer-span input or automatic candidate answer-span
    extraction.
-   Automatic question generation using T5 with beam-search decoding.
-   Three model options:
    1.  **Online Pre-Trained** --- loads `valhalla/t5-small-qg-hl` from
        the Hugging Face Hub.
    2.  **Local Pre-Trained** --- loads the same pre-trained model from
        `saved_pre_trained_model/`.
    3.  **Local Trained/Fine-Tuned** --- loads the fine-tuned model from
        `saved_model/`.
-   Quantitative evaluation using BLEU-4, METEOR, ROUGE-1, and ROUGE-L.
-   Qualitative manual evaluation of answerability, relevance, and
    linguistic fluency.
-   CSV export of generated questions and answer spans.
-   Streamlit web interface.
-   Completed fine-tuning and held-out test-set evaluation.

The final fine-tuning experiment used a 20,000-record subset of SQuAD
v1.1. The experiment was executed in a CUDA-enabled Kaggle environment
using two NVIDIA Tesla T4 GPUs with FP16 enabled. The 20,000 records
were divided into 16,000 training records, 2,000 validation records, and
2,000 held-out test records.

------------------------------------------------------------------------

## 2. Repository Structure

``` text
Group238/
│
├── app.py                         # Streamlit web application
├── train-gpu.ipynb                # Executed Kaggle GPU fine-tuning notebook
├── model_utils.py                 # Inference, answer extraction and evaluation
├── README.md                      # Setup and execution instructions
├── saved_pre_trained_model/       # Local copy of the pre-trained model
├── saved_model/                   # Fine-tuned model and tokenizer
├── GPU_Fine_Tuning_Artifacts/     # Completed Kaggle training evidence
│   ├── loss_curve.png
│   ├── test_results.json
│   ├── train-gpu.pdf
│   └── training_log.csv
├── sample_inputs/                 # Sample study-material inputs
└── requirements.txt               # Python dependencies
```
------------------------------------------------------------------------

## 3. System Requirements

### Python Version

The project was developed and tested using Python 3.11.x for the local
application environment.

Python installed version can be check by:

``` bash
python --version
```

On Windows:

``` bash
py --version
```

### Hardware

The Streamlit application supports CPU and CUDA-enabled GPU execution
for inference when a compatible PyTorch installation is available.

The final fine-tuning experiment was performed using the `train-gpu.ipynb` notebook in a Kaggle GPU environment with two NVIDIA Tesla T4 GPUs (2 × 14.6 GB VRAM) and FP16 enabled.

-   2 × NVIDIA Tesla T4 GPUs
-   FP16 enabled
-   20,000 SQuAD v1.1 records
-   3 epochs

### Internet Connection

Internet access is required for:

-   Initial dependency installation (pip install -r requirements.txt).
-   Downloading the Pre-Trained and Fine-Tuned model archive from GitHub when the local model folders are not available.
-   Loading the Online Pre-Trained model from Hugging Face.
-   Downloading required NLTK tokenizer resources (punkt, wordnet).

The Local Pre-Trained and Local Trained/Fine-Tuned options can use the
model files stored locally.

------------------------------------------------------------------------

## 4. Dataset Specification

The project uses the **Stanford Question Answering Dataset (SQuAD)
v1.1**.

  ---------------------------------------------------------------------------------
  Field                               Details
  ----------------------------------- ---------------------------------------------
  Dataset Name                        Stanford Question Answering Dataset (SQuAD)
                                      v1.1

  Source / Repository                 Hugging Face Datasets --- `rajpurkar/squad`

  Original Dataset Homepage           Stanford SQuAD ---
                                      https://rajpurkar.github.io/SQuAD-explorer/

  Dataset License                     Creative Commons Attribution-ShareAlike 4.0
                                      International (CC BY-SA 4.0)

  Original Dataset Size               87,599 training examples and 10,570
                                      validation examples

  Size Used in Project                20,000 records selected from the SQuAD v1.1
                                      training split
  ---------------------------------------------------------------------------------

### Final Dataset Split

``` text
Total records : 20,000
Training      : 16,000
Validation    : 2,000
Held-out Test : 2,000
```

Split ratio:

``` text
80% Training
10% Validation
10% Held-out Test
```

The fine-tuned model generated from this experiment is already available for download through the following GitHub Release:

[Download Pre-Trained and Fine-Tuned Models](https://github.com/2025aa05353-sudo/NLP-Assignment-2-Models/releases/download/v1.0.0/NLP_Assignment-2_Models.zip)

The ZIP archive contains:

```text
saved_pre_trained_model/
saved_model/
```
------------------------------------------------------------------------

## 5. Model Description

The project uses:

``` text
valhalla/t5-small-qg-hl
```

This is a pre-trained T5-small Transformer Encoder--Decoder (Seq2Seq)
model adapted for question generation.

### Model Architecture

  Item                          Description
  ----------------------------- -----------------------------------
  Model                         T5-small
  Full Name                     Text-to-Text Transfer Transformer
  Architecture                  Transformer
  Encoder--Decoder              Yes
  Seq2Seq                       Yes
  LSTM                          No
  Task                          Automatic Question Generation
  Question Generation Variant   `qg`
  Highlighted Answer Span       `hl`

### Model Name Explanation

  Part    Meaning
  ------- ----------------------------------------
  T5      Text-to-Text Transfer Transformer
  small   Smaller version of the T5 architecture
  qg      Question Generation
  hl      Highlighted answer span

### Model Files

The application provides three model sources:

#### 1. Online Pre-Trained

The application loads:

``` text
valhalla/t5-small-qg-hl
```

directly from the Hugging Face Hub.

#### 2. Local Pre-Trained

The same pre-trained model is stored locally in:

``` text
saved_pre_trained_model/
```

The folder contains the model and tokenizer files required for local
loading.

#### 3. Local Trained/Fine-Tuned

The fine-tuned model is stored in:

``` text
saved_model/
```

This model was fine-tuned using the selected 20,000 SQuAD v1.1 records in the Kaggle GPU environment.

All three options use the same T5-small Transformer Encoder--Decoder
(Seq2Seq) architecture.

------------------------------------------------------------------------

## 6. Project Setup

Copy the project files to the local machine.

The project directory should contain the application files and, when
available, the local model directories:

``` text
app.py
model_utils.py
requirements.txt
README.md
saved_pre_trained_model/
saved_model/
GPU_Fine_Tuning_Artifacts/
sample_inputs/
train-gpu.ipynb
```

------------------------------------------------------------------------

## 7. Create a Virtual Environment

A virtual environment is recommended to isolate project dependencies.

### Windows

Open Command Prompt or Anaconda Prompt in the project directory:

``` bash
py -3.11 -m venv .venv
```

Activate it:

``` bash
.venv\Scripts\activate
```

### Linux / macOS

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
```

------------------------------------------------------------------------

## 8. Install Dependencies

Upgrade pip:

``` bash
python -m pip install --upgrade pip
```

Install project dependencies:

``` bash
python -m pip install -r requirements.txt
```

Verify Streamlit:

``` bash
python -m streamlit --version
```

Verify PyTorch:

``` bash
python -c "import torch; print(torch.__version__)"
```

------------------------------------------------------------------------

## 9. Model Availability and Download

The project contains two local model directories:

```text
saved_pre_trained_model/
saved_model/
```

### Local Pre-Trained Model

The `saved_pre_trained_model/` directory contains the original pre-trained:

```text
valhalla/t5-small-qg-hl
```

model and tokenizer.

### Local Fine-Tuned Model

The `saved_model/` directory contains the model and tokenizer produced by the `train-gpu.ipynb` notebook during the Kaggle GPU fine-tuning experiment using the selected 20,000 SQuAD v1.1 records.

### GitHub Hosted Model Weights

Because the model directories are large, the prepared model weights are hosted as a downloadable archive through the project's configured GitHub Releases link.


[Download Pre-Trained and Fine-Tuned Models](https://github.com/2025aa05353-sudo/NLP-Assignment-2-Models/releases/download/v1.0.0/NLP_Assignment-2_Models.zip)

The archive contains:

```text
saved_pre_trained_model/
saved_model/
```

If either local model directory is missing, use the **Download & Extract Model Weights from GitHub** option provided in the Streamlit application.

The application checks whether the local model directories already exist. If they are available, they can be used directly. If they are missing, the application provides the GitHub-hosted model archive download option.

> **Important:** The Streamlit application does not perform model fine-tuning. Fine-tuning was completed separately in the Kaggle GPU environment before the final application was prepared.

---

## 10. Final Fine-Tuning Experiment — Kaggle

The final model was fine-tuned in a Kaggle GPU environment using:

```text
train-gpu.ipynb
```

The executed notebook is preserved as:

```text
GPU_Fine_Tuning_Artifacts/train-gpu.pdf
```

The final experiment used:

| Parameter | Value |
|---|---|
| Base model | `valhalla/t5-small-qg-hl` |
| Dataset | SQuAD v1.1 |
| Total records | 20,000 |
| Training records | 16,000 |
| Validation records | 2,000 |
| Held-out test records | 2,000 |
| Epochs | 3 |
| Per-device batch size | 16 |
| GPUs | 2 × NVIDIA Tesla T4 |
| FP16 | Enabled |
| Learning rate | `5e-5` |
| Optimizer | AdamW |
| Warmup steps | 300 |
| Random seed | 42 |
| Maximum input length | 512 tokens |
| Maximum target length | 64 tokens |
| Evaluation frequency | Every 50 optimization steps |

The completed training pipeline generated:

```text
GPU_Fine_Tuning_Artifacts/
├── loss_curve.png
├── test_results.json
├── train-gpu.pdf
└── training_log.csv
```

The executed notebook documents the SQuAD v1.1 loading, preprocessing, 20,000-record subset, 80/10/10 data split, model fine-tuning, validation, held-out test evaluation, and artifact generation.

### Final Evaluation Result

```text
Best Validation Loss : 3.0153
Final Validation Loss: 3.0163
Held-Out Test Loss   : 3.0279
Test Samples         : 2,000
```

The execution summary recorded in the training run was:

```text
Device        : CUDA GPU (Tesla T4) x2
Samples       : 20,000
Epochs        : 3
Total Duration: 23m 58s
Test Loss     : 3.0279
```

### Important Reproducibility Note

The Kaggle fine-tuning run is provided as **completed execution evidence and documentation**.

The final Streamlit application does **not** start or execute a fine-tuning pipeline. The application uses the already-created local checkpoints, or downloads the prepared model archive from GitHub when the local checkpoints are not present.

---

## 11. Kaggle Training Evidence in the Application

The Streamlit application provides a dedicated section for the completed Kaggle training experiment.

### GPU Cloud Training Pipeline

The application presents information from:

```text
train-gpu.ipynb
```

including the dual-GPU execution configuration and training summary.

### Executed GPU Notebook PDF

The application provides a viewer and download option for:

```text
GPU_Fine_Tuning_Artifacts/train-gpu.pdf
```

This PDF contains the executed notebook and its cell-by-cell training output.

### Training Metrics

The application displays the generated:

```text
GPU_Fine_Tuning_Artifacts/loss_curve.png
GPU_Fine_Tuning_Artifacts/training_log.csv
GPU_Fine_Tuning_Artifacts/test_results.json
```

These artifacts document the training/validation loss progression and held-out test evaluation.

---

## 12. Training and Validation Loss

The final Kaggle run records training and validation loss every 50 optimization steps.

The generated loss graph is:

```text
GPU_Fine_Tuning_Artifacts/loss_curve.png
```

The graph shows the training and validation cross-entropy loss over the optimization steps.

Key reported values are:

- Best validation loss: **3.0153**
- Final validation loss: **3.0163**
- Held-out test loss: **3.0279**
- Final optimization step: **1,500**

The validation loss remained close to its best value toward the end of the three-epoch run.

---

## 13. Training Artifacts

The `GPU_Fine_Tuning_Artifacts/` folder contains the evidence generated from the completed Kaggle training execution.

### `train-gpu.pdf`

A PDF copy of the executed Kaggle notebook containing the training code and execution output.

### `loss_curve.png`

The step-level training and validation loss graph.

### `training_log.csv`

The recorded training and validation loss values.

### `test_results.json`

The held-out test-set result and final experiment configuration.

### `saved_model/`

The resulting fine-tuned model and tokenizer used by the application.

---

## 14. Launch the Streamlit Application

Activate the project environment and run:

``` bash
streamlit run app.py
```

Alternatively:

``` bash
python -m streamlit run app.py
```

The application is normally available at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 15. Using the Application

### Step 1 --- Select a Model

The application provides three model options:

``` text
Online Pre-Trained
Local Pre-Trained
Local Trained/Fine-Tuned
```

#### Online Pre-Trained

Loads:

``` text
valhalla/t5-small-qg-hl
```

from the Hugging Face Hub.

#### Local Pre-Trained

Loads:

``` text
saved_pre_trained_model/
```

from the local project directory.

#### Local Trained/Fine-Tuned

Loads:

``` text
saved_model/
```

from the local project directory.

For the final assignment demonstration, the **Local Trained/Fine-Tuned**
option can be used when `saved_model/` is available.

### Step 2 --- Provide Study Material

The application supports:

-   Sample study material.
-   Direct text input.
-   `.txt` upload.
-   `.pdf` upload.
-   `.csv` upload.

### Step 3 --- Provide an Answer Span

An answer span may be entered manually.

Examples:

``` text
TCP
```

or:

``` text
1939 to 1945
```

If an answer span is not supplied, the application can attempt automatic
candidate answer-span extraction.

### Step 4 --- Generate Questions

Configure the question-generation settings and select:

``` text
Generate Question(s)
```

The generated questions are displayed together with their corresponding
answer spans.

### Step 5 --- Download the Quiz

Generated questions can be exported as a CSV file using the
application's download option.

------------------------------------------------------------------------

## 16. Supported Input Formats

### Direct Text

Study material can be entered directly into the application.

### TXT

Upload a plain-text `.txt` document.

### PDF

Upload a PDF containing machine-readable text.

Scanned/image-only PDFs may not provide usable text for question
generation.

### CSV

Upload a `.csv` document containing study material.

------------------------------------------------------------------------

## 17. Evaluation

The application provides automatic evaluation metrics:

-   BLEU-4
-   METEOR
-   ROUGE-1
-   ROUGE-L

### Evaluation Procedure

1.  Generate a question.
2.  Provide the generated question to the evaluation interface.
3.  Provide the corresponding human-written reference question.
4.  Run the evaluation.
5.  Review the displayed metric scores.

### Manual Answerability and Relevance Assessment

Generated questions should also be evaluated qualitatively.

For each selected question, check:

-   **Answerability:** Can the question be answered from the supplied
    passage?
-   **Factually Relevant:** Is the question relevant to the information
    in the passage?
-   **Linguistic Fluency:** Is the question grammatically clear and
    natural?

------------------------------------------------------------------------

## 18. Demonstration Subject Areas

The application includes examples from different subject areas,
including:

### Computer Networks

Example topic:

``` text
TCP / Transmission Control Protocol
```

### World War II History

Example topic:

``` text
World War II
```

These examples demonstrate the use of the question-generation system
across different study-material domains.

------------------------------------------------------------------------

## 19. Model Loading and Local Checkpoint Behaviour

The local model-loading implementation performs explicit checkpoint
validation.

For a requested local model directory:

-   The directory must exist.
-   The directory must not be empty.
-   The model is loaded from the specified local directory.
-   A missing local checkpoint raises an error instead of silently
    switching to an online model.

Therefore, when using:

``` text
saved_pre_trained_model/
```

or:

``` text
saved_model/
```

the corresponding directory must be available.

The **Online Pre-Trained** option is separate and intentionally loads
the Hugging Face model online.

------------------------------------------------------------------------

## 20. Known Issues and Limitations

### 20.1 Fine-Tuning Time

The final fine-tuning experiment was completed on Kaggle using two NVIDIA
Tesla T4 GPUs and 20,000 SQuAD v1.1 records for three epochs.

The executed Kaggle notebook and generated training artifacts are provided
for documentation and evaluation evidence.

The Streamlit application does not perform fine-tuning.

### 20.2 Internet Connection

Internet access is required for:

- Loading the Online Pre-Trained model.
- Downloading the Pre-Trained and Fine-Tuned model archive from GitHub when
  the local model folders are not available.
- Downloading NLTK resources.

The local model options can work from locally stored checkpoints.

### 20.3 PDF Text Extraction

The application requires machine-readable text from PDF files. Scanned
or image-only PDFs may not produce usable text.

### 20.4 NLTK Resources

The application initializes the required NLTK resources. If the
execution environment blocks downloads, the resources may need to be
installed manually.

### 20.5 Model Storage

The `saved_pre_trained_model/` and `saved_model/` directories are 
large (approximately 424 MB in total) because they contain model weights.

### 20.6 Dataset Size

The final reported experiment uses a 20,000-record subset of SQuAD v1.1
rather than the complete training set. Increasing the dataset size or
number of epochs will increase computational time and storage
requirements.

### 20.7 Long Passages

Context inputs exceeding 512 tokens are truncated according to the
configured maximum input length.
------------------------------------------------------------------------

## 21. Quick Start --- Application

### Windows

``` bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

### Linux / macOS

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 22. If a Local Model Is Not Available

If `saved_pre_trained_model/` or `saved_model/` is not present in the project directory:

1. Open the Streamlit application.
2. Go to the model-management section.
3. Use **Download & Extract Model Weights from GitHub**.
4. The prepared model archive is downloaded and extracted.
5. The required local model directories become available for inference.

The application does **not** retrain the model during this process.

The fine-tuned model was already created in the Kaggle GPU experiment and is preserved as:

```text
saved_model/
```

If the local folders are already present, no download is required.

## 23. Final Output

The completed project provides:

1. A Streamlit-based automatic question-generation application.
2. Online Pre-Trained model access.
3. Local Pre-Trained model access.
4. Local Trained/Fine-Tuned model access.
5. SQuAD v1.1-based fine-tuning using 20,000 records.
6. Completed Kaggle dual-GPU training evidence.
7. Training and validation loss tracking.
8. Held-out test-set evaluation.
9. BLEU-4, METEOR, ROUGE-1 and ROUGE-L evaluation.
10. Manual answerability, relevance and fluency assessment.
11. CSV quiz export.
12. GitHub-hosted model-weight recovery when local model folders are unavailable.

The final Streamlit application uses the already-trained model checkpoints for question generation. The Kaggle notebook, executed PDF, loss curve, training log, and test results are retained as evidence of the completed fine-tuning experiment.
