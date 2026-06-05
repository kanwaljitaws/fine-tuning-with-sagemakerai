# Fine-Tuning LLMs for Financial Stock Forecasting on Amazon SageMaker

Fine-tune **Qwen3-4B-Instruct** on the [FinGPT/fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) dataset using QLoRA on Amazon SageMaker AI, then evaluate and deploy the model as a real-time endpoint.

## Use Case

Given a company's recent news and financial metrics, the fine-tuned model produces structured analyst-style output:
- **Positive Developments** — bullish signals
- **Potential Concerns** — bearish / risk signals
- **Prediction** — directional stock movement for the upcoming week (e.g. *up by 3–4%*)

Dataset covers **Dow Jones 30** stocks from **May 2023 – April 2024** (1,230 training examples).

---

## End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Your Local Machine / Studio                  │
│                                                                     │
│   FinGPT Dataset (HuggingFace Hub)                                  │
│        │  load_dataset()                                            │
│        ▼                                                            │
│   Format as chat messages  →  train/test JSON  →  Upload to S3      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Amazon S3 (model + data storage)                 │
│                                                                     │
│   s3://YOUR-BUCKET/                                                 │
│   ├── datasets/llm-fine-tuning-fingpt-forecaster/                   │
│   │   ├── train/dataset.json                                        │
│   │   └── test/dataset.json                                         │
│   ├── models/Qwen_Qwen3-4B-Instruct-2507/   ← base model weights   │
│   └── train-Qwen3-4B-Instruct-sft-script/   ← fine-tuned output    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SageMaker Training Job Architecture                    │
│                                                                     │
│  ModelTrainer.train()                                               │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────────────────────────────────┐                        │
│  │        ml.g5.2xlarge instance           │                        │
│  │        (NVIDIA A10G · 24 GB VRAM)       │                        │
│  │                                         │                        │
│  │  PyTorch 2.6 Training Container         │                        │
│  │  ┌───────────────────────────────────┐  │                        │
│  │  │  train.py                         │  │  Input Channels:       │
│  │  │                                   │  │  /opt/ml/input/data/   │
│  │  │  1. Load model from S3            │◄─┼── train/dataset.json   │
│  │  │  2. Load tokenizer                │  │  /opt/ml/input/data/   │
│  │  │  3. Apply 4-bit NF4 quant         │◄─┼── test/dataset.json    │
│  │  │  4. Add LoRA adapters             │  │  /opt/ml/input/data/   │
│  │  │     (all linear layers, r=8)      │◄─┼── config/args.yaml     │
│  │  │  5. SFTTrainer (1 epoch)          │  │                        │
│  │  │  6. Merge LoRA → base model       │  │  Output:               │
│  │  │  7. Save merged weights           │──┼► /opt/ml/model/        │
│  │  └───────────────────────────────────┘  │       │                │
│  └─────────────────────────────────────────┘       │                │
│                                                     ▼                │
│                                            Uploaded to S3            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SageMaker Real-time Endpoint                           │
│                                                                     │
│  ml.g5.2xlarge  ·  DJL LMI Container (vLLM backend)                │
│                                                                     │
│  POST /invocations                                                  │
│  { "messages": [...], "parameters": { "max_new_tokens": 1024 } }    │
│        │                                                            │
│        ▼                                                            │
│  [Positive Developments]: ...                                       │
│  [Potential Concerns]: ...                                          │
│  [Prediction & Analysis]: up by 3-4%                                │
└─────────────────────────────────────────────────────────────────────┘
```

### SageMaker Training Job — Data Flow

```
S3 Input Channels                Container paths          Script reads via
─────────────────                ───────────────          ───────────────
train/dataset.json    ────────►  /opt/ml/input/data/train/    args.train_dataset_path
test/dataset.json     ────────►  /opt/ml/input/data/test/     args.test_dataset_path
args.yaml             ────────►  /opt/ml/input/data/config/   TrlParser
Base model (S3/HF)    ────────►  /tmp/tmp_folder/             snapshot_download / aws s3 cp

Trained model         ◄────────  /opt/ml/model/               trainer.save_model()
(auto-uploaded to S3 by SageMaker after job completes)
```

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| AWS Account | With SageMaker, S3, and IAM permissions |
| SageMaker Studio v2 | JupyterLab 4 environment |
| Notebook instance | `ml.t3.medium` or larger |
| Training instance | `ml.g5.2xlarge` (NVIDIA A10G, 24 GB VRAM) |
| Inference instance | `ml.g5.2xlarge` |
| Python | 3.10+ |

---

## Step 0 — Create a SageMaker Studio Domain and User Profile

If you don't have a SageMaker Studio domain yet, follow these steps.

### Option A — Quick setup via AWS Console (recommended for first-time users)

1. Go to **AWS Console** → search **SageMaker** → click **Studio**
2. Click **Set up for single user (Quick)** if this is a new account
   - AWS creates a domain and user profile automatically with a default IAM role
3. Click **Open Studio** — this opens SageMaker Studio v2 (JupyterLab 4)

> Quick setup takes ~5 minutes on first launch.

---

### Option B — Custom domain setup (recommended for teams / production)

#### 1. Create an IAM Role for SageMaker

In **IAM Console** → **Roles** → **Create role**:
- Trusted entity: **AWS service → SageMaker**
- Attach policies:
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
- Name it: `SageMakerStudioRole`
- Copy the **Role ARN** (you'll need it below)

#### 2. Create the Studio Domain

In **SageMaker Console** → **Domains** → **Create domain**:

| Field | Value |
|-------|-------|
| Name | `my-studio-domain` (any name) |
| Authentication | IAM |
| Default execution role | `SageMakerStudioRole` (created above) |
| VPC | Default VPC (or your custom VPC) |
| Subnets | Select at least one subnet |
| App network access type | `PublicInternetOnly` (for simplest setup) |

Click **Submit** — domain creation takes ~5–10 minutes.

#### 3. Create a User Profile

In **SageMaker Console** → **Domains** → click your domain → **Add user**:

| Field | Value |
|-------|-------|
| Name | your username (e.g. `workshop-user`) |
| Execution role | `SageMakerStudioRole` |

Click **Submit**.

#### 4. Open Studio

In **Domains** → click your domain → click your user → **Launch** → **Studio**

This opens **SageMaker Studio v2** with JupyterLab 4.

#### 5. Select a JupyterLab Space (Studio v2)

Inside Studio, click **JupyterLab** in the left sidebar → **Create JupyterLab Space**:

| Field | Value |
|-------|-------|
| Name | `workshop-space` |
| Instance | `ml.t3.medium` (for running notebooks) |
| Storage | 50 GB |
| Image | `SageMaker Distribution 2.x` |

Click **Run Space** → **Open JupyterLab**.

---

## Step 1 — Clone this Repository

Inside JupyterLab, open a **Terminal** (File → New → Terminal):

```bash
cd ~
git clone https://github.com/kanwaljitaws/fine-tuning-with-sagemakerai.git
cd fine-tuning-with-sagemakerai
```

---

## Step 2 — Run the Notebooks in Order

Open each notebook and run all cells: **Run → Run All Cells**

---

### Task 1 — Deploy the Base Model
📓 `task_01_foundation_model_playground/01.01_search_and_deploy_huggingface_llm.ipynb`

**What it does:**
- Downloads Qwen3-4B-Instruct locally → syncs to S3
- Deploys base model to a SageMaker endpoint (LMI / DJL container)
- Tests a financial forecasting prompt
- Stores `BASE_ENDPOINT_NAME`

> ⏱ ~5–8 min for endpoint to become `InService`

---

### Task 2 — Fine-Tune with QLoRA
📓 `task_02_customize_foundation_model/02.01_finetune_Qwen3-4B-instruct.ipynb`

**What it does:**
- Loads FinGPT dataset (`num_samples=100` by default; increase to `1230` for full training)
- Formats data as chat messages → uploads to S3
- Writes `args.yaml` with hyperparameters
- Launches SageMaker Training Job (QLoRA on `ml.g5.2xlarge`)
- Merges LoRA adapter into base model
- Deploys fine-tuned model to a new endpoint
- Stores `TUNED_ENDPOINT_NAME`

> ⏱ Training (~90 samples): ~5–10 min  
> ⏱ Endpoint creation: ~5–8 min

**Key hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_seq_length` | 1500 | Covers 95% of FinGPT sequence lengths |
| `lora_r` / `lora_alpha` | 8 / 16 | LoRA rank |
| `per_device_train_batch_size` | 2 | Fits A10G with gradient checkpointing |
| `gradient_accumulation_steps` | 2 | Effective batch size = 4 |
| `num_train_epochs` | 1 | Increase for better results |
| `fp16` | true | Halves memory vs FP32 |
| `merge_weights` | true | No PEFT needed at inference |

---

### Task 3 — Evaluate with ROUGE Scores
📓 `task_03_foundation_model_evaluation/03.01_foundation_model_evaluation_lighteval.ipynb`

**What it does:**
- Loads 10 samples from FinGPT test split
- Queries both endpoints with identical prompts
- Computes ROUGE-1, ROUGE-2, ROUGE-L vs ground-truth answers
- Plots comparison bar chart
- Shows side-by-side predictions

---

### Task 4 — Responsible AI (optional)
📓 `task_04_responsible_ai/04.01_bedrock_guardrails_apply_guardrail_api.ipynb`

Apply Amazon Bedrock Guardrails to filter harmful content.

---

### Task 5 — FMOps Pipeline (optional)
📓 `task_05_fmops/05.01_fine-tuning-pipeline.ipynb`

Automate the full workflow as a **SageMaker Pipeline** with MLflow experiment tracking.

---

## Step 3 — Clean Up

Run the cleanup cells at the end of Task 3, or manually:

```python
import boto3
client = boto3.client('sagemaker')
client.delete_endpoint(EndpointName=BASE_ENDPOINT_NAME)
client.delete_endpoint(EndpointName=TUNED_ENDPOINT_NAME)
```

To delete the Studio Space when done: **Studio → JupyterLab → Stop Space**.

---

## Estimated Costs

| Resource | Approx. cost |
|----------|-------------|
| `ml.g5.2xlarge` training (~10 min) | ~$0.08 |
| `ml.g5.2xlarge` endpoint (per hour each) | ~$1.52/hr |
| S3 storage (~8 GB model) | ~$0.18/month |
| `ml.t3.medium` JupyterLab Space (per hour) | ~$0.05/hr |

> ⚠️ Delete endpoints and stop Spaces when not in use.

---

## Repository Structure

```
├── task_01_foundation_model_playground/
│   └── 01.01_search_and_deploy_huggingface_llm.ipynb
├── task_02_customize_foundation_model/
│   ├── 02.01_finetune_Qwen3-4B-instruct.ipynb
│   ├── scripts/train.py
│   └── scripts/requirements.txt
├── task_03_foundation_model_evaluation/
│   └── 03.01_foundation_model_evaluation_lighteval.ipynb
├── task_04_responsible_ai/
│   └── 04.01_bedrock_guardrails_apply_guardrail_api.ipynb
├── task_05_fmops/
│   └── 05.01_fine-tuning-pipeline.ipynb
└── utilities/
    └── helpers.py
```

---

## Dataset

[FinGPT/fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405)

| Split | Examples |
|-------|----------|
| train | 1,230 |
| test | 300 |

Columns: `prompt`, `answer`, `period`, `label`, `symbol`
