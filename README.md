# Fine-Tuning LLMs for Financial Stock Forecasting on Amazon SageMaker

Fine-tune **Qwen3-4B-Instruct** on the [FinGPT/fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) dataset using QLoRA on Amazon SageMaker AI, then evaluate and deploy the model as a real-time endpoint.

## Use Case

Given a company's recent news and financial metrics, the fine-tuned model produces structured analyst-style output:
- **Positive Developments** — bullish signals
- **Potential Concerns** — bearish / risk signals
- **Prediction** — directional stock movement for the upcoming week (e.g. *up by 3–4%*)

Dataset covers **Dow Jones 30** stocks from **May 2023 – April 2024** (1,230 training examples).

---

## Architecture

```
FinGPT Dataset (HuggingFace)
    │  format as chat messages
    ▼
Amazon S3  (train + test splits)
    │
    ▼
SageMaker Training Job  ← QLoRA (4-bit + LoRA) on ml.g5.2xlarge
  Qwen3-4B-Instruct + LoRA adapter → merged model → S3
    │
    ▼
SageMaker Real-time Endpoint  (LMI / DJL container)
    │
    ▼
Financial forecast inference
```

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| AWS Account | With SageMaker, S3, and IAM permissions |
| SageMaker Studio v2 | JupyterLab 4 environment |
| Instance for notebooks | `ml.t3.medium` or larger |
| Instance for training | `ml.g5.2xlarge` (NVIDIA A10G, 24 GB VRAM) |
| Instance for inference | `ml.g5.2xlarge` |
| Python | 3.10+ |

---

## Step-by-Step Instructions

### 1. Open SageMaker Studio v2

1. Sign in to the [AWS Console](https://console.aws.amazon.com/sagemaker)
2. Navigate to **Amazon SageMaker** → **Studio**
3. Click **Open Studio** on your user profile (ensure you are using **Studio v2 / JupyterLab 4**)

### 2. Clone this repository

Open a **Terminal** in Studio (File → New → Terminal) and run:

```bash
cd ~
git clone https://github.com/kanwaljitaws/fine-tuning-with-sagemakerai.git
cd fine-tuning-with-sagemakerai
```

### 3. Run the notebooks in order

Open each notebook in JupyterLab and run all cells top to bottom (**Run → Run All Cells**).

---

#### Task 1 — Deploy Base Model
📓 `task_01_foundation_model_playground/01.01_search_and_deploy_huggingface_llm.ipynb`

- Downloads **Qwen3-4B-Instruct** locally and syncs it to S3
- Deploys the **base model** to a SageMaker real-time endpoint using the LMI container
- Runs a test financial forecasting prompt
- Saves `BASE_ENDPOINT_NAME` for use in later notebooks

> ⏱ Endpoint creation: ~5–8 minutes

---

#### Task 2 — Fine-Tune with QLoRA
📓 `task_02_customize_foundation_model/02.01_finetune_Qwen3-4B-instruct.ipynb`

- Loads the **FinGPT forecaster** dataset (100 samples by default; set `num_samples=1230` for full dataset)
- Formats examples as chat messages (`system` / `user` / `assistant`)
- Uploads train and test splits to S3
- Writes `args.yaml` with training hyperparameters
- Launches a **SageMaker Training Job** (QLoRA, 1 epoch, `ml.g5.2xlarge`)
- Merges LoRA adapter into base model and saves to S3
- Deploys the **fine-tuned model** to a new endpoint
- Saves `TUNED_ENDPOINT_NAME` for evaluation

> ⏱ Training job (~90 samples): ~5–10 minutes  
> ⏱ Endpoint creation: ~5–8 minutes

**Key hyperparameters** (`args.yaml`):

| Parameter | Value |
|-----------|-------|
| `max_seq_length` | 1500 |
| `lora_r` / `lora_alpha` | 8 / 16 |
| `per_device_train_batch_size` | 2 |
| `num_train_epochs` | 1 |
| `fp16` | true |
| `merge_weights` | true |

---

#### Task 3 — Evaluate
📓 `task_03_foundation_model_evaluation/03.01_foundation_model_evaluation_lighteval.ipynb`

- Loads 10 random samples from the FinGPT **test** split
- Sends identical prompts to both the base and fine-tuned endpoints
- Computes **ROUGE-1, ROUGE-2, ROUGE-L** scores against ground-truth answers
- Plots a comparison bar chart
- Prints side-by-side predictions for qualitative review

---

#### Task 4 — Responsible AI (optional)
📓 `task_04_responsible_ai/04.01_bedrock_guardrails_apply_guardrail_api.ipynb`

Apply Amazon Bedrock Guardrails to filter harmful content from model outputs.

---

#### Task 5 — FMOps Pipeline (optional)
📓 `task_05_fmops/05.01_fine-tuning-pipeline.ipynb`

Automate the full fine-tune → evaluate → register → deploy workflow as a **SageMaker Pipeline** with MLflow experiment tracking.

---

### 4. Clean up endpoints

At the end of Task 3 there are cleanup cells that delete both endpoints. Run them to avoid ongoing charges.

To manually delete:
```python
import boto3
client = boto3.client('sagemaker')
client.delete_endpoint(EndpointName=BASE_ENDPOINT_NAME)
client.delete_endpoint(EndpointName=TUNED_ENDPOINT_NAME)
```

---

## Estimated Costs

| Resource | Approx. cost |
|----------|-------------|
| `ml.g5.2xlarge` training (~10 min) | ~$0.08 |
| `ml.g5.2xlarge` endpoint (per hour) | ~$1.52/hr |
| S3 storage (model ~8 GB) | ~$0.18/month |

> Delete endpoints when not in use to avoid charges.

---

## Repository Structure

```
├── task_01_foundation_model_playground/
│   └── 01.01_search_and_deploy_huggingface_llm.ipynb   # Deploy base model
├── task_02_customize_foundation_model/
│   ├── 02.01_finetune_Qwen3-4B-instruct.ipynb          # QLoRA fine-tuning
│   ├── scripts/train.py                                 # Training script
│   └── scripts/requirements.txt
├── task_03_foundation_model_evaluation/
│   └── 03.01_foundation_model_evaluation_lighteval.ipynb  # ROUGE evaluation
├── task_04_responsible_ai/
│   └── 04.01_bedrock_guardrails_apply_guardrail_api.ipynb
├── task_05_fmops/
│   └── 05.01_fine-tuning-pipeline.ipynb                # SageMaker Pipeline
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
