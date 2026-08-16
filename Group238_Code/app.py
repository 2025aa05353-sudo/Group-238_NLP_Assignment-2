"""
================================================================================
File Name   : app.py
Purpose     : Streamlit Web Application Interface for Automatic Question 
              Generation, GPU Training Pipeline Showcase & PDF Viewer,
              Checkpoint Engine Management, and BLEU/METEOR/ROUGE Evaluation.
              Includes GitHub Releases model weights download and extraction.
================================================================================
Course      : Natural Language Processing - Assignment 2
Program     : M.Tech. in AIML, BITS Pilani (WILP)
Group No    : Group 238
================================================================================
"""

import os
import sys
import time
import json
import zipfile
import base64
from pathlib import Path
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from model_utils import QuestionGenerator, compute_metrics

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Automatic Question Generation - Group 238",
    page_icon="📚",
    layout="wide"
)

# ==============================================================================
# GITHUB RELEASES WEIGHTS DOWNLOAD HELPER (DYNAMIC STREAMING & SAFE EXTRACTION)
# ==============================================================================
GITHUB_DOWNLOAD_URL = "https://github.com/2025aa05353-sudo/NLP-Assignment-2-Models/releases/download/v1.0.0/NLP_Assignment-2_Models.zip"
ZIP_NAME = "NLP_Assignment-2_Models.zip"

def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Validates that extracted files remain within the destination root directory."""
    try:
        base = base_dir.resolve()
        target = target_path.resolve()
        return base in target.parents or base == target
    except Exception:
        return False

def download_and_extract_weights():
    """
    Downloads model weights archive directly from GitHub Releases with live chunk streaming,
    real-time progress updates, zip integrity verification, and safe directory extraction.
    """
    import requests

    base_dir = Path(__file__).resolve().parent
    zip_dest = base_dir / ZIP_NAME

    # Reset previous notifications and set lock
    st.session_state["download_notification"] = None
    st.session_state["is_downloading"] = True

    # Clean up leftover files from previous partial attempts
    if zip_dest.exists():
        try:
            os.remove(str(zip_dest))
        except OSError:
            pass

    progress_bar = st.progress(0)
    status_header = st.empty()
    status_percent = st.empty()
    timer_badge = st.empty()

    try:
        # ----------------------------------------------------------------------
        # STAGE 1: STREAM DOWNLOAD DIRECTLY WITH LIVE CHUNK PROGRESS
        # ----------------------------------------------------------------------
        status_header.markdown("⏳ **Stage 1/2: Downloading model weights from GitHub Releases...**")
        status_percent.markdown("🔄 *Connecting to GitHub CDN...*")
        progress_bar.progress(5)
        start_time = time.time()

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        response = session.get(GITHUB_DOWNLOAD_URL, stream=True, timeout=45)
        
        if response.status_code != 200:
            raise ConnectionError(f"GitHub server returned HTTP status code {response.status_code}. Please verify release visibility.")

        header_len = response.headers.get('content-length')
        total_expected_bytes = int(header_len) if header_len and header_len.isdigit() else 445063168

        downloaded_bytes = 0
        chunk_size = 1024 * 1024  # 1 MB chunk

        with open(zip_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    elapsed = int(time.time() - start_time)
                    mins, secs = divmod(elapsed, 60)
                    total_mb = total_expected_bytes / (1024 * 1024)
                    curr_mb = downloaded_bytes / (1024 * 1024)
                    timer_badge.caption(f"⏱️ **Elapsed Time:** `{mins}m {secs:02d}s` (Downloading ~{total_mb:.1f} MB archive)")

                    pct = min(int((downloaded_bytes / total_expected_bytes) * 80), 80)
                    download_pct = min(int((downloaded_bytes / total_expected_bytes) * 100), 100)

                    progress_bar.progress(max(pct, 5))
                    status_percent.markdown(
                        f"📥 **Download Status:** `{download_pct}%` complete (`{curr_mb:.1f} MB` / `{total_mb:.1f} MB`)"
                    )

        progress_bar.progress(85)
        status_percent.markdown(f"📥 **Download Status:** `100%` complete (`{downloaded_bytes / (1024*1024):.1f} MB` received)")

        # ----------------------------------------------------------------------
        # STAGE 2: LIVE EXTRACTION WITH ZIP INTEGRITY TEST & PATH VALIDATION
        # ----------------------------------------------------------------------
        status_header.markdown("📦 **Stage 2/2: Extracting model folders (`saved_model/`, `saved_pre_trained_model/`)...**")

        with zipfile.ZipFile(str(zip_dest), "r") as zip_ref:
            corrupted_file = zip_ref.testzip()
            if corrupted_file is not None:
                raise ConnectionError(f"Downloaded archive is corrupted at member: '{corrupted_file}'.")

            members = zip_ref.infolist()
            total_files = len(members)

            for idx, member in enumerate(members, start=1):
                destination_path = base_dir / member.filename
                if not is_safe_path(base_dir, destination_path):
                    raise SecurityError(f"Blocked path-traversal in archive member: '{member.filename}'")
                
                zip_ref.extract(member, str(base_dir))
                extract_pct = int((idx / total_files) * 100)
                total_bar_pct = 85 + int((idx / total_files) * 15)

                progress_bar.progress(min(total_bar_pct, 100))
                status_percent.markdown(f"📦 **Extraction Status:** `{extract_pct}%` (`{idx}/{total_files}` files extracted)")
                time.sleep(0.02)

        # Store persistent success notification in session state
        st.session_state["download_notification"] = {
            "type": "success",
            "message": "✅ **Model weights downloaded (100%) and extracted successfully! Model engines are now ready for inference.**"
        }
        st.cache_resource.clear()
        st.session_state["is_downloading"] = False
        st.rerun()

    except Exception as e:
        status_header.empty()
        status_percent.empty()
        timer_badge.empty()
        progress_bar.empty()

        # Store persistent error notification in session state
        st.session_state["download_notification"] = {
            "type": "error",
            "message": f"❌ **Connection Interrupted / Download Failed:** {str(e)}",
            "help": "⚠️ The partial download was discarded. Please verify your internet connection and try again."
        }
        st.session_state["is_downloading"] = False

    finally:
        if zip_dest.exists():
            try:
                os.remove(str(zip_dest))
            except OSError:
                pass


# ==============================================================================
# SESSION STATE INITIALIZATION (DEFAULT: LOCAL CUSTOM FINE-TUNED)
# ==============================================================================
if "candidate_q" not in st.session_state:
    st.session_state["candidate_q"] = "What is the main function of TCP?"

if "selected_model_path" not in st.session_state:
    st.session_state["selected_model_path"] = "./saved_model"

if "model_type_label" not in st.session_state:
    st.session_state["model_type_label"] = "Local Fine-Tuned (./saved_model)"

if "is_downloading" not in st.session_state:
    st.session_state["is_downloading"] = False

if "download_notification" not in st.session_state:
    st.session_state["download_notification"] = None

if "generated_results_cache" not in st.session_state:
    st.session_state["generated_results_cache"] = None


# ==============================================================================
# SIDEBAR NAVIGATION & CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.title("📌 Group 238 - NLP Assignment 2 Navigation")
    app_page = st.radio(
        "Select Page View:",
        [
            "👥 Student & Group Info",
            "⚙️ Model Selection & Training Evidence", 
            "📚 Automatic Question Generation for E-Learning"
        ]
    )
    
    st.divider()

    # Show Model Configuration ONLY on the Question Generation Page
    if app_page == "📚 Automatic Question Generation for E-Learning":
        st.header("⚙️ Model Configuration")
        num_beams = st.slider("Beam Search Width", min_value=1, max_value=4, value=2, step=1)
        num_return_seqs = st.slider("Candidate Questions Count", min_value=1, max_value=num_beams, value=1, step=1)
        limit_qa = st.slider("Max Automatic Q&A Pairs", min_value=1, max_value=5, value=5, step=1)
        
        st.caption(f"**Active Engine:** {st.session_state['model_type_label']}")
    else:
        num_beams = 2
        num_return_seqs = 1
        limit_qa = 5


# ==============================================================================
# MODEL INITIALIZATION (STRICT LOCAL CHECKPOINT LOADING)
# ==============================================================================
@st.cache_resource
def load_generator(model_path):
    """
    Loads QuestionGenerator for the requested model path.
    Returns None if loading fails or local files are missing, ensuring clean error reporting.
    """
    try:
        return QuestionGenerator(model_name=model_path)
    except Exception as e:
        return None


# ==============================================================================
# PAGE 1: STUDENT & GROUP INFO PAGE
# ==============================================================================
if app_page == "👥 Student & Group Info":
    st.title("👥 (Group 238) Student Details & Group Contributions")
    st.caption("Natural Language Processing | Assignment - 2 | Group 238")
    st.markdown("---")

    st.subheader("👨‍💻 Group# 238 Members & Contribution Matrix")

    member_data = [
        {"ID": "2025AA05353", "Name": "SHEETAL PRAKASH BARANWAL", "Contribution": "100%"},
        {"ID": "2025AA05174", "Name": "AISHWARYA TEWARI", "Contribution": "100%"},
        {"ID": "2025AA05002", "Name": "JADHAV VEDANT BHAGWAN SWATI", "Contribution": "100%"},
        {"ID": "2025AB05063", "Name": "J RAJESH KUMAR", "Contribution": "100%"},
        {"ID": "2025AB05120", "Name": "JAHNAVI KOLLIPARA", "Contribution": "100%"},
    ]

    st.table(pd.DataFrame(member_data))

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📚 Academic Context")
        st.markdown("""
        * **Program:** M.Tech. in Artificial Intelligence & Machine Learning
        * **Institution:** BITS Pilani (Work Integrated Learning Programmes)
        * **Course:** Natural Language Processing
        * **Assignment:** Assignment - 2 (Automatic Question Generation Engine)
        * **Group Identification:** Group 238
        """)

    with col2:
        st.subheader("⚙️ System Specifications")
        st.markdown("""
        * **Base Model Architecture:** `valhalla/t5-small-qg-hl`
        * **Domain Adaptation:** Fine-tuned on SQuAD v1.1 (20,000 records subset)
        * **Frameworks:** PyTorch, Hugging Face Transformers, NLTK, Streamlit
        * **Deployment:** CPU / CUDA GPU Support
        * **Evaluation Suite:** BLEU-4, METEOR, ROUGE-1 (F1), ROUGE-L (F1)
        """)


# ==============================================================================
# PAGE 2: MODEL SELECTION & TRAINING EVIDENCE
# ==============================================================================
elif app_page == "⚙️ Model Selection & Training Evidence":
    st.title("⚙️ Group 238 - Model Engine Selection & Management")
    st.caption("Natural Language Processing | Assignment - 2 | Group 238")

    # Quick 4-Line Overview Index
    st.markdown("""
    **What are in this Page:**
    1. **Active Engine Selection:** Toggle inference backend among one of Model selection : Custom Fine-Tuned Model OR Local Pre-Trained OR Remote Hugging Face Hub.
    2. **GitHub Model Weights Downloader:** Direct streaming download and extraction of pre-trained and fine-tuned model packages.
    3. **Dual-GPU Training & PDF Showcase:** Dual Tesla T4 execution summary, live training terminal logs, and `train-gpu.ipynb` (as PDF) viewer/download.
    4. **Evaluation Artifacts:** Step-level loss curve graph (`loss_curve.png`), training logs table (`training_log.csv`), and held-out test metrics.
    """)
    st.markdown("---")

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "GPU_Fine_Tuning_Artifacts"
    
    saved_fine_tuned = base_dir / "saved_model"
    saved_pre_trained = base_dir / "saved_pre_trained_model"

    fine_tuned_exists = saved_fine_tuned.exists() and saved_fine_tuned.is_dir() and len(os.listdir(saved_fine_tuned)) > 0
    pre_trained_exists = saved_pre_trained.exists() and saved_pre_trained.is_dir() and len(os.listdir(saved_pre_trained)) > 0

    # --------------------------------------------------------------------------
    # 1. MODEL CHECKPOINT SELECTION RADIO
    # --------------------------------------------------------------------------
    st.subheader("1. Select Active Model Weights Engine")
    
    options = [
        "Local Custom Fine-Tuned Model (./saved_model)",
        "Local Pre-trained Model (./saved_pre_trained_model)",
        "Hugging Face Remote Hub (valhalla/t5-small-qg-hl)"
    ]

    current_path = st.session_state.get("selected_model_path", "")
    if current_path == str(saved_pre_trained) or current_path == "./saved_pre_trained_model":
        idx = 1
    elif current_path == "valhalla/t5-small-qg-hl":
        idx = 2
    else:
        idx = 0

    model_option = st.radio("Choose Model Checkpoint for Inference:", options, index=idx)

    if model_option == "Local Custom Fine-Tuned Model (./saved_model)":
        st.session_state["selected_model_path"] = str(saved_fine_tuned)
        st.session_state["model_type_label"] = "Local Fine-Tuned (./saved_model)"
        if fine_tuned_exists:
            st.success("✅ Active Engine: Local custom fine-tuned weights loaded!")
        else:
            st.error("❌ **Local Fine-Tuned model (`./saved_model`) is not available on disk.** Please download and extract the weights using Option 2 below before running inference.")

    elif model_option == "Local Pre-trained Model (./saved_pre_trained_model)":
        st.session_state["selected_model_path"] = str(saved_pre_trained)
        st.session_state["model_type_label"] = "Local Pre-Trained (./saved_pre_trained_model)"
        if pre_trained_exists:
            st.success("✅ Active Engine: Local pre-trained model folder loaded!")
        else:
            st.error("❌ **Local Pre-Trained model files not found on disk.** Please download and extract them using Option 2 below.")

    else:
        st.session_state["selected_model_path"] = "valhalla/t5-small-qg-hl"
        st.session_state["model_type_label"] = "Remote Hub (valhalla/t5-small-qg-hl)"
        st.info("ℹ️ Active Engine: Base model loaded directly from Hugging Face Hub.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. GITHUB RELEASES HOSTED WEIGHTS DOWNLOAD CARD
    # --------------------------------------------------------------------------
    with st.expander("☁️ **2. Download GitHub Hosted Model Weights on Local system (Taxila < 10 MB Limit)**", expanded=True):
        st.markdown(f"""
        To comply with Taxila assignment portal upload limits (< 10 MB), the model weights archive (`NLP_Assignment-2_Models.zip` ~424 MB) is hosted on GitHub Releases:
        * 🔗 **Direct Archive Link:** [{GITHUB_DOWNLOAD_URL}]({GITHUB_DOWNLOAD_URL})
        * 📦 **Archive Contents:** Both `./saved_pre_trained_model` and `./saved_model` weight directories.
        """)

        # Real-time Folder Status Badges
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            if fine_tuned_exists:
                st.success("✅ `saved_model/` (fine-tuned) is present locally.")
            else:
                st.warning("⚠️ `saved_model/` (fine-tuned) is missing locally.")
        with col_status2:
            if pre_trained_exists:
                st.success("✅ `saved_pre_trained_model/` is present locally.")
            else:
                st.warning("⚠️ `saved_pre_trained_model/` is missing locally.")

        st.write("")

        # Render Persistent Status Notification
        notif = st.session_state.get("download_notification", None)
        if notif:
            if notif["type"] == "success":
                st.success(notif["message"])
            elif notif["type"] == "error":
                st.error(notif["message"])
                if "help" in notif:
                    st.warning(notif["help"])

        is_busy = st.session_state.get("is_downloading", False)

        # Conditional Warning & Refresh Confirmation
        if pre_trained_exists and fine_tuned_exists:
            st.warning("⚠️ **Notice:** Both model weight directories already exist locally on disk. Re-downloading will overwrite existing files.")
            confirm_refresh = st.checkbox("Confirm: I want to overwrite and refresh existing local model files", disabled=is_busy)

            if st.button("📥 Refresh & Re-download Model Weights from GitHub", type="primary", width="stretch", disabled=is_busy):
                if confirm_refresh:
                    download_and_extract_weights()
                else:
                    st.warning("⚠️ Please check the confirmation box above if you wish to refresh and re-download the model weights.")
        else:
            if st.button("📥 Download & Extract Model Weights from GitHub", type="primary", width="stretch", disabled=is_busy):
                download_and_extract_weights()

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. MANAGEMENT TABS: GPU NOTEBOOK SHOWCASE & PDF VIEWER
    # --------------------------------------------------------------------------
    tab1, tab2 = st.tabs([
        "📓 GPU Cloud Training Pipeline (Kaggle T4 x2)", 
        "📄 Executed GPU Notebook (train-gpu.pdf)"
    ])

    # TAB 1: GPU Cloud Training Pipeline (train-gpu.ipynb Showcase)
    with tab1:
        st.subheader("📓 3. Dual-GPU Fine-Tuning Pipeline (`train-gpu.ipynb`)")
        st.write("Complete end-to-end training execution log from the high-performance Kaggle GPU environment.")

        # Responsive 4-Column Metric Cards (No text truncation)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hardware", "Dual Tesla T4", "2 × 14.6 GB VRAM", delta_color="off")
        m2.metric("Dataset Subset", "20,000 Samples", "80 / 10 / 10 Split", delta_color="off")
        m3.metric("Training Time", "23m 27s", "3 Full Epochs", delta_color="off")
        m4.metric("Held-Out Test Loss", "3.0279", "Evaluated on 2,000 test samples", delta_color="off")

        st.write("")
        st.markdown("#### ⚡ Experiment Execution Summary")
        col_nb1, col_nb2 = st.columns(2)
        with col_nb1:
            st.markdown("""
            * **Architecture:** `valhalla/t5-small-qg-hl` (Encoder-Decoder)
            * **Dataset:** SQuAD v1.1 (`rajpurkar/squad`)
            * **Splits:** 16,000 Train (80%) | 2,000 Val (10%) | 2,000 Test (10%)
            * **Input / Target Length:** 512 tokens / 64 tokens
            """)
        with col_nb2:
            st.markdown("""
            * **Optimizer:** AdamW (`adamw_torch`) with FP16 enabled
            * **Learning Rate Schedule:** $5 \\times 10^{-5}$ with 10% Warmup (300 steps)
            * **Total Optimization Steps:** 3,000 steps (1,000 steps/epoch)
            * **Exported Artifact:** `Group238_final_training_artifacts.zip` (212.44 MB)
            """)

        with st.expander("📜 **View Live GPU Execution Output (Kaggle Terminal Log)**", expanded=False):
            st.code("""
==============================================================================
[System Init] Automatic device detection
[System Init] Device      : CUDA GPU (Tesla T4) x2
[System Init] CUDA        : True
[System Init] GPU count   : 2
[System Init] Batch size  : 16
[System Init] FP16        : True
[System Init] Samples     : 20000
[System Init] Epochs      : 3
==============================================================================
[GPU 0] Tesla T4 | VRAM: 14.56 GB
[GPU 1] Tesla T4 | VRAM: 14.56 GB

[Model] Loading local pre-trained model...
[Model] Path: /kaggle/working/saved_pre_trained_model
[Timing] Model/tokenizer load: 0.7s
[1/4] Loading SQuAD v1.1 (sample subset size: 20000)...
[Dataset Split] Train: 16000 | Validation: 2000 | Test: 2000
[Timing] Dataset preparation: 13.4s

[Training Configuration]
Training samples       : 16000
Validation samples     : 2000
Test samples           : 2000
Epochs                 : 3
Batch size             : 16
Steps per epoch        : 1000
Total training steps   : 3000
Learning rate          : 5e-05
Warmup ratio           : 0.1
Warmup steps           : 300
Input max length       : 512
Target max length      : 64

[2/4] Starting model fine-tuning process...
[Training Step 500] Epoch 1.0  | Training Loss: 3.4486 | Validation Loss: 3.0305
[Training Step 1000] Epoch 2.0 | Training Loss: 3.4391 | Validation Loss: 3.0166
[Training Step 1500] Epoch 3.0 | Training Loss: 3.2887 | Validation Loss: 3.0163
[Timing] Fine-tuning duration: 23m 27s

[2.5/4] Evaluating fine-tuned model on held-out Test Set...
[Test Set Metrics] Held-Out Test Loss: 3.0279
[Artifact] Held-out test evaluation metrics saved to 'test_results.json'

[3/4] Saving model weights and tokenizer...
[System Check] Model weights successfully saved to 'saved_model'

[4/4] Generating training artifacts...
[Artifact] Unified training/validation metrics log saved to 'training_log.csv'
[Artifact] Step-level loss graph exported to 'loss_curve.png'
==============================================================================
[Execution Summary] Device: CUDA GPU (Tesla T4) x2 | Samples: 20000 | Epochs: 3 | Total Duration: 23m 58s | Test Loss: 3.0279
==============================================================================
            """, language="text")

    # TAB 2: Executed GPU Notebook PDF Viewer & Download Option
    with tab2:
        st.subheader("📄 Executed GPU Notebook Report (`train-gpu.pdf`)")
        st.write("View the complete, cell-by-cell execution output from the dual-GPU training notebook or download the original PDF.")

        pdf_path = artifacts_dir / "train-gpu.pdf" if (artifacts_dir / "train-gpu.pdf").exists() else base_dir / "train-gpu.pdf"

        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
                base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            col_dl, _ = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="📥 Download Executed Notebook (PDF)",
                    data=pdf_bytes,
                    file_name="train-gpu.pdf",
                    mime="application/pdf",
                    type="primary"
                )

            st.write("")
            pdf_iframe = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_iframe, unsafe_allow_html=True)
        else:
            st.warning("⚠️ `train-gpu.pdf` not found in `GPU_Fine_Tuning_Artifacts/` or workspace root.")

    # --------------------------------------------------------------------------
    # 4. ARTIFACTS SECTION
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Training Metrics & Loss Curve Artifacts")
    col_img, col_csv = st.columns(2)

    loss_img_path = artifacts_dir / "loss_curve.png" if (artifacts_dir / "loss_curve.png").exists() else base_dir / "loss_curve.png"
    log_csv_path = artifacts_dir / "training_log.csv" if (artifacts_dir / "training_log.csv").exists() else base_dir / "training_log.csv"
    test_json_path = artifacts_dir / "test_results.json" if (artifacts_dir / "test_results.json").exists() else base_dir / "test_results.json"

    with col_img:
        st.markdown("**Loss Curve Graph (`loss_curve.png`)**")
        if loss_img_path.exists():
            st.image(str(loss_img_path), caption="Step-Level Training and Validation Loss Plot", width="stretch")
        else:
            st.info("Loss plot artifact `loss_curve.png` is generated during training.")

    with col_csv:
        st.markdown("**Training Metrics & Test Evaluation**")
        if test_json_path.exists():
            try:
                with open(test_json_path, "r", encoding="utf-8") as tf:
                    t_data = json.load(tf)
                    st.success(f"🎯 **Held-Out Test Set Loss:** `{t_data.get('test_loss', '3.0279')}` (Evaluated on {t_data.get('eval_samples', 2000)} samples)")
            except Exception:
                pass

        if log_csv_path.exists():
            try:
                st.dataframe(pd.read_csv(log_csv_path), width="stretch")
            except Exception as e:
                st.error(f"Error loading CSV log: {str(e)}")
        else:
            st.info("Training metrics artifact `training_log.csv` will appear here.")


# ==============================================================================
# PAGE 3: AUTOMATIC QUESTION GENERATION FOR E-LEARNING
# ==============================================================================
else:
    st.title("📚 Group 238 - Automatic Question Generation for E-Learning")
    st.caption("Natural Language Processing | Assignment - 2 | Group 238")
    st.markdown("---")

    qg_engine = load_generator(st.session_state["selected_model_path"])

    # Strict Validation: Stop execution and warn evaluator if weights are missing
    if qg_engine is None:
        st.error(
            f"❌ **Active Model Engine Unavailable:** The selected engine checkpoint (`{st.session_state['model_type_label']}`) "
            f"is missing on disk.\n\n"
            f"👉 **Action Required:** Please navigate to **⚙️ Model Selection & Training Evidence** and click **Download & Extract Model Weights from GitHub** "
            f"to populate the local weight directories."
        )
        st.info("💡 Alternatively, you can select the **Hugging Face Remote Hub (valhalla/t5-small-qg-hl)** option for instant cloud inference without local files.")
        st.stop()
    else:
        st.info(f"🤖 **Current Active Model Engine:** `{st.session_state['model_type_label']}`")

    # 1. INPUT CONTEXT PASSAGE
    st.subheader("1. Input Context Passage")

    input_mode = st.radio(
        "Select Input Source Mode:",
        ["📋 Sample Inputs", "✍️ Direct Text Input", "📄 Document Upload (.txt, .pdf, .csv)"],
        horizontal=True
    )

    context_text = ""

    if input_mode == "📋 Sample Inputs":
        sample_options = {
            "-- Select a Sample --": "",
            "Sample 1: Computer Networks (CS)": (
                "The Transmission Control Protocol (TCP) is one of the main protocols of the Internet protocol suite. "
                "It originated in the initial network implementation in which it complemented the Internet Protocol (IP). "
                "TCP provides reliable, ordered, and error-checked delivery of a stream of octets between applications "
                "running on hosts communicating via an IP network. Major Internet applications such as the World Wide Web, "
                "email, remote administration, and file transfer rely on TCP."
            ),
            "Sample 2: World War History": (
                "World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's "
                "countries including all of the great powers forming two opposing military alliances: the Allies and the Axis. "
                "The Empire of Japan aimed to dominate Asia and the Pacific, while Nazi Germany sought to conquer Europe."
            )
        }

        selected_sample = st.selectbox("Choose Sample Material:", list(sample_options.keys()))
        if selected_sample != "-- Select a Sample --":
            context_text = sample_options[selected_sample]
            st.info(context_text)

    elif input_mode == "✍️ Direct Text Input":
        direct_input = st.text_area("Paste passage or study material:", height=180, placeholder="Paste your paragraph here...")
        if direct_input.strip():
            context_text = direct_input.strip()

    elif input_mode == "📄 Document Upload (.txt, .pdf, .csv)":
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "csv"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".txt"):
                    context_text = uploaded_file.read().decode("utf-8", errors="replace")
                elif uploaded_file.name.endswith(".pdf"):
                    pdf_reader = PdfReader(uploaded_file)
                    extracted_pages = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_pages.append(page_text)
                    context_text = "\n".join(extracted_pages)
                elif uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    context_text = "\n".join(df.astype(str).values.flatten())
                
                if context_text.strip():
                    st.success(f"File '{uploaded_file.name}' loaded successfully!")
                else:
                    st.warning("Uploaded file contained no extractable text.")
            except Exception as e:
                st.error(f"❌ File Parsing Exception: {str(e)}")

    # Context Token Length Notice (T5 512 Max Token Capacity)
    if context_text.strip():
        estimated_token_count = len(context_text.split())
        if estimated_token_count > 450:
            st.warning(
                f"ℹ️ **Context Length Notice:** The input context contains ~{estimated_token_count} words. "
                "T5 sequence generation processes up to 512 tokens; excess content will be truncated for question generation."
            )

    # 2. TARGET ANSWER & QUESTION GENERATION
    st.markdown("---")
    st.subheader("2. Target Answer & Question Generation")

    col1, col2 = st.columns([1, 1])

    with col1:
        target_answer = st.text_input("Optional Target Answer Span (e.g., 'TCP' or '1939 to 1945'):", placeholder="Leave blank for automatic extraction")

    with col2:
        st.write(" ")
        st.write(" ")
        generate_btn = st.button("⚡ Generate Question(s)", type="primary", width="stretch")

    if generate_btn:
        if not context_text.strip():
            st.warning("Please select a sample, paste text, or upload a document first!")
        else:
            if target_answer.strip():
                with st.spinner("Generating targeted question candidate(s)..."):
                    try:
                        gen_output = qg_engine.generate(
                            context=context_text, 
                            answer=target_answer, 
                            num_beams=num_beams, 
                            num_return_sequences=num_return_seqs
                        )
                        
                        if isinstance(gen_output, list):
                            st.session_state["candidate_q"] = gen_output[0]
                            rows = []
                            for idx_q, q_item in enumerate(gen_output, start=1):
                                rows.append({
                                    "Context": context_text,
                                    "Target Answer": target_answer,
                                    "Candidate Number": idx_q,
                                    "Generated Question": q_item
                                })
                            export_df = pd.DataFrame(rows)
                        else:
                            st.session_state["candidate_q"] = gen_output
                            export_df = pd.DataFrame([{
                                "Context": context_text, 
                                "Target Answer": target_answer, 
                                "Candidate Number": 1, 
                                "Generated Question": gen_output
                            }])
                        
                        st.session_state["generated_results_cache"] = {
                            "mode": "candidates",
                            "data": gen_output,
                            "df": export_df
                        }
                    except Exception as e:
                        st.error(f"❌ Question Generation Exception: {str(e)}")
            else:
                with st.spinner("Parsing text and extracting target Q&A pairs..."):
                    try:
                        results = qg_engine.extract_spans_and_generate(context_text, limit=limit_qa, num_beams=num_beams)
                        
                        if results:
                            st.session_state["candidate_q"] = results[0][2]
                            data = []
                            for ctx, ans, q in results:
                                data.append({"Context Sentence": ctx, "Target Answer": ans, "Generated Question": q})
                            df_results = pd.DataFrame(data)
                            
                            st.session_state["generated_results_cache"] = {
                                "mode": "table",
                                "df": df_results
                            }
                        else:
                            st.session_state["generated_results_cache"] = None
                            st.info("No distinct noun entities found. Try typing a specific target answer above.")
                    except Exception as e:
                        st.error(f"❌ Span Extraction & Generation Exception: {str(e)}")

    # Persistent Display of Generated Results (Remains visible during evaluation)
    cached_res = st.session_state.get("generated_results_cache")
    if cached_res:
        st.markdown("### Generated Results")
        if cached_res["mode"] == "candidates":
            gen_out = cached_res["data"]
            if isinstance(gen_out, list):
                st.success("**Generated Candidate Questions:**")
                for idx_q, q_item in enumerate(gen_out, start=1):
                    st.write(f"❓ **Candidate {idx_q}:** {q_item}")
            else:
                st.success("**Generated Question:**")
                st.write(f"❓ {gen_out}")
            
            csv_data = cached_res["df"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Generated Quiz (CSV)",
                data=csv_data,
                file_name="generated_quiz.csv",
                mime="text/csv",
                key="btn_dl_candidates"
            )
        elif cached_res["mode"] == "table":
            st.table(cached_res["df"])
            csv_data = cached_res["df"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Generated Quiz (CSV)",
                data=csv_data,
                file_name="generated_quiz.csv",
                mime="text/csv",
                key="btn_dl_table"
            )

    # 3. EVALUATION ENGINE (BLEU, METEOR & ROUGE SCORES)
    st.markdown("---")
    st.subheader("3. Evaluation Engine (BLEU, METEOR & ROUGE Scores)")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        cand_q = st.text_input("Candidate / Generated Question:", value=st.session_state["candidate_q"])
    with eval_col2:
        ref_q = st.text_input(
            "Ideal Human Reference Question:", 
            value="", 
            placeholder="Enter human reference question here to do the evaluation"
        )

    if st.button("📊 Compute Evaluation Metrics"):
        if ref_q.strip() and cand_q.strip():
            try:
                metrics = compute_metrics(ref_q, cand_q)
                
                metrics_df = pd.DataFrame(
                    list(metrics.items()), 
                    columns=["Evaluation Metric", "Score"]
                )
                
                table_col, _ = st.columns([1, 1])
                with table_col:
                    st.table(metrics_df)
                
                bleu4 = metrics.get("BLEU-4", 0.0)
                meteor = metrics.get("METEOR", 0.0)
                rouge1 = metrics.get("ROUGE-1 (F1)", 0.0)
                rougel = metrics.get("ROUGE-L (F1)", 0.0)
                
                bleu_status = "High" if bleu4 >= 0.25 else "Low"
                meteor_status = "High" if meteor >= 0.40 else "Low"
                rouge1_status = "High" if rouge1 >= 0.50 else "Low"
                rougel_status = "High" if rougel >= 0.50 else "Low"
                
                st.markdown("#### 📌 Score-Based Observations & Analysis")
                
                # BLEU-4 Observation
                if bleu_status == "High":
                    bleu_obs = "**High Precision:** Candidate question matches exact multi-word phrases and sequence order from the human reference."
                else:
                    bleu_obs = "**Low Precision:** Exact n-gram match is low due to alternate phrasing, syntactic variation, or synonym substitution."

                # METEOR Observation (Stem, Synonym & Paraphrase Analysis)
                if meteor_status == "High":
                    meteor_obs = "**High Semantic & Paraphrase Alignment:** Captures key conceptual meanings through effective word stem matches and WordNet synonym alignments, maintaining strong chunk order."
                else:
                    meteor_obs = "**Low Paraphrase Overlap:** Significant lexical divergence from the reference phrasing, missing core morphological stems or synonym mappings."

                # ROUGE-1 Observation
                if rouge1_status == "High":
                    rouge1_obs = "**High Unigram Recall:** Core subject entities and essential topic keywords from the reference are well preserved."
                else:
                    rouge1_obs = "**Low Unigram Recall:** Key domain terms from the ideal reference question are absent."

                # ROUGE-L Observation
                if rougel_status == "High":
                    rougel_obs = "**High Structural Fluency:** Longest common sentence structure and word ordering closely match human gold-standard writing."
                else:
                    rougel_obs = "**Low Clause Alignment:** Sentence structure or clause arrangement differs noticeably from the reference question."

                st.markdown(f"""
                * **BLEU-4 ({bleu4:.4f} - {bleu_status}):** {bleu_obs}
                * **METEOR ({meteor:.4f} - {meteor_status}):** {meteor_obs}
                * **ROUGE-1 ({rouge1:.4f} - {rouge1_status}):** {rouge1_obs}
                * **ROUGE-L ({rougel:.4f} - {rougel_status}):** {rougel_obs}
                * **Decoding Strategy:** Beam search decoding ($k = {num_beams}$) actively mitigates repetitive token loops during generation.
                """)
            except Exception as e:
                st.error(f"❌ Evaluation Computation Exception: {str(e)}")
        else:
            st.warning("Please enter an ideal human reference question above to compute metrics.")