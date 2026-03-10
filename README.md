# whisper-gpu-worker-build

AWS Batch worker that downloads a pre-converted WAV file from S3, transcribes it with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), formats an HTML transcript,
summarises it with OpenAI, and POSTs the result to the Legislata API.

A **Lambda orchestrator** dispatches each job through a 4-tier fallback system so that
a missing GPU spot instance never leaves the job stuck RUNNABLE for hours.

---

## Architecture

```
SQS Queue
    │
    ▼ (batch size 1, ReportBatchItemFailures)
Lambda Orchestrator  ──── EventBridge Scheduler (30-min one-shot per job)
    │                              │
    │  Tier 1: GPU spot            │ check → still stuck? escalate
    │  Tier 2: GPU on-demand       │
    │  Tier 3: CPU spot            │
    │  Tier 4: CPU on-demand       │
    ▼
AWS Batch Job  (worker.py reads JOB_PAYLOAD env var)
    └── S3 download → faster-whisper → OpenAI summarise → POST to Legislata API
```

### Fallback flow

| Tier | Queue | Job Definition | Instance types |
|------|-------|----------------|---------------|
| 1 | `whisper-gpu-spot-queue` | `whisper-gpu` | g4dn.xlarge / g4dn.2xlarge (SPOT) |
| 2 | `whisper-gpu-ondemand-queue` | `whisper-gpu` | same (ON_DEMAND) |
| 3 | `whisper-cpu-spot-queue` | `whisper-cpu` | c5.4xlarge / m5.4xlarge (SPOT) |
| 4 | `whisper-cpu-ondemand-queue` | `whisper-cpu` | same (ON_DEMAND) |

The Lambda checks job status every 30 minutes. If the job is `RUNNABLE`/`PENDING`/`STARTING`
it cancels it and escalates to the next tier. If the job `SUCCEEDED` it deletes the SQS
message. If the job `FAILED` at tier < 4 it immediately escalates. At tier 4 (CPU
on-demand) the check is rescheduled indefinitely — CPU on-demand will always eventually
run.

---

## AWS Resources to Create Manually

### Compute Environments (4)

| Name | Type | Instance Types | Min vCPUs | Max vCPUs |
|------|------|---------------|-----------|-----------|
| `whisper-gpu-spot` | MANAGED / SPOT / EC2 | `g4dn.xlarge`, `g4dn.2xlarge` | 0 | 8 |
| `whisper-gpu-ondemand` | MANAGED / ON_DEMAND / EC2 | `g4dn.xlarge`, `g4dn.2xlarge` | 0 | 8 |
| `whisper-cpu-spot` | MANAGED / SPOT / EC2 | `c5.4xlarge`, `m5.4xlarge` | 0 | 32 |
| `whisper-cpu-ondemand` | MANAGED / ON_DEMAND / EC2 | `c5.4xlarge`, `m5.4xlarge` | 0 | 32 |

All compute environments need an EC2 instance role with the
`AmazonEC2ContainerServiceforEC2Role` managed policy attached.

### Job Queues (4)

Each queue points to its single compute environment with priority 1.

| Queue Name | Compute Environment |
|-----------|-------------------|
| `whisper-gpu-spot-queue` | `whisper-gpu-spot` |
| `whisper-gpu-ondemand-queue` | `whisper-gpu-ondemand` |
| `whisper-cpu-spot-queue` | `whisper-cpu-spot` |
| `whisper-cpu-ondemand-queue` | `whisper-cpu-ondemand` |

### Job Definitions (2)

Both use the same ECR image (`whisper-gpu-worker:latest` / `:cpu`); the device is
selected by the `WHISPER_DEVICE` env var.

**`whisper-gpu`**
- Image: `<account>.dkr.ecr.us-east-2.amazonaws.com/whisper-gpu-worker:latest`
- vCPU: 4 | Memory: 14000 MB
- Resource requirements: `{"type": "GPU", "value": "1"}`
- Env vars: `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`
- Secrets (from Secrets Manager / Parameter Store): `OPENAI_API_KEY`, `LEGISLATA_API_AUTH_KEY`

**`whisper-cpu`**
- Image: `<account>.dkr.ecr.us-east-2.amazonaws.com/whisper-gpu-worker:cpu`
- vCPU: 8 | Memory: 16000 MB
- No GPU resource requirement
- Env vars: `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`
- Same secrets as above

### Lambda (`whisper-orchestrator`)

- Runtime: Python 3.11
- Handler: `lambda_orchestrator.lambda_handler`
- Timeout: 60 seconds | Memory: 256 MB
- SQS trigger: transcription queue, batch size 1, `ReportBatchItemFailures` enabled
- Environment variables:

  | Variable | Description |
  |----------|-------------|
  | `ESCALATION_SCHEDULE_ROLE_ARN` | IAM role ARN EventBridge Scheduler uses to invoke this Lambda |
  | `LAMBDA_ARN` | This Lambda's own ARN |
  | `SQS_QUEUE_URL` | Source SQS queue URL |
  | `AWS_REGION` | Region (default `us-east-2`) |

- IAM permissions required:
  - `batch:SubmitJob`, `batch:DescribeJobs`, `batch:CancelJob`
  - `sqs:DeleteMessage`, `sqs:ChangeMessageVisibility`
  - `scheduler:CreateSchedule`, `scheduler:DeleteSchedule`
  - `iam:PassRole` (scoped to the EventBridge Scheduler role)

### EventBridge Scheduler Role

- Trust policy principal: `scheduler.amazonaws.com`
- Permission: `lambda:InvokeFunction` on the Lambda ARN

---

## Environment Variables (worker.py)

| Variable | Default | Description |
|----------|---------|-------------|
| `JOB_PAYLOAD` | — | JSON job payload set by Lambda/Batch (primary input) |
| `OPENAI_API_KEY` | **required** | OpenAI API key |
| `LEGISLATA_API_AUTH_KEY` | **required** | Legislata API auth key |
| `POSTS_URL` | `https://legislata-backend-production.herokuapp.com/public/api/v1/posts` | Post API endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `WHISPER_MODEL` | `small` | faster-whisper model size |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | e.g. `float16`, `int8` |
| `WHISPER_BEAM_SIZE` | `1` | Beam size for transcription |
| `TURN_GAP_SECONDS` | `1.2` | Silence gap (seconds) that starts a new paragraph |
| `VAD_FILTER` | `true` | Enable voice-activity-detection filter |
| `WORD_TIMESTAMPS` | `true` | Enable word-level timestamps |
| `SUMMARY_CHUNK_CHARS` | `4000` | Max chars per OpenAI summarisation chunk |
| `SUMMARY_SECTION_TOKENS` | `200` | Max tokens per section summary |
| `MAX_SUMMARY_TOKENS` | `500` | Max tokens for final summary |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | `us-east-2` | AWS region |

---

## Local Testing

```bash
# Pass payload as a command-line argument
python3 worker.py '{"video_id": "t5pBl8-xBd8", "url": "...", "title": "...", "office_id": "1672", "s3_bucket": "...", "s3_key": "..."}'

# Or via stdin
echo '{"video_id": "t5pBl8-xBd8", ...}' | python3 worker.py

# Or via environment variable (mimics Batch behaviour)
export JOB_PAYLOAD='{"video_id": "t5pBl8-xBd8", ...}'
python3 worker.py
```

Required environment variables for local testing:
```bash
export OPENAI_API_KEY=sk-...
export LEGISLATA_API_AUTH_KEY=...
export AWS_REGION=us-east-2
# Optionally override device:
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
```

