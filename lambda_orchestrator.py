"""
Lambda orchestrator for Whisper GPU/CPU Batch job 4-tier fallback.

Triggered by:
  1. SQS messages (event['source'] == 'sqs' is implied by Records presence)
  2. EventBridge Scheduler one-time schedules (event['source'] == 'escalation')

4-tier fallback:
  Tier 1: GPU spot        (whisper-gpu-spot-queue,     whisper-gpu job def)
  Tier 2: GPU on-demand   (whisper-gpu-ondemand-queue, whisper-gpu job def)
  Tier 3: CPU spot        (whisper-cpu-spot-queue,     whisper-cpu job def)
  Tier 4: CPU on-demand   (whisper-cpu-ondemand-queue, whisper-cpu job def)

Required Lambda env vars:
  ESCALATION_SCHEDULE_ROLE_ARN  IAM role ARN that EventBridge Scheduler uses to invoke this Lambda
  LAMBDA_ARN                    This Lambda's own ARN (for scheduling targets)
  SQS_QUEUE_URL                 Source SQS queue URL (for deleting messages on completion)
  AWS_REGION / AWS_DEFAULT_REGION  default: us-east-2
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIER_MAP = {
    1: {"queue": "whisper-gpu-spot-queue",     "job_def": "whisper-gpu",  "label": "GPU spot"},
    2: {"queue": "whisper-gpu-ondemand-queue", "job_def": "whisper-gpu",  "label": "GPU on-demand"},
    3: {"queue": "whisper-cpu-spot-queue",     "job_def": "whisper-cpu",  "label": "CPU spot"},
    4: {"queue": "whisper-cpu-ondemand-queue", "job_def": "whisper-cpu",  "label": "CPU on-demand"},
}

REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")
LAMBDA_ARN = os.environ.get("LAMBDA_ARN", "")
ESCALATION_SCHEDULE_ROLE_ARN = os.environ.get("ESCALATION_SCHEDULE_ROLE_ARN", "")

# ---------------------------------------------------------------------------
# AWS clients (module-level for Lambda connection re-use)
# ---------------------------------------------------------------------------

_batch = None
_scheduler = None
_sqs = None


def get_batch():
    global _batch
    if _batch is None:
        _batch = boto3.client("batch", region_name=REGION)
    return _batch


def get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = boto3.client("scheduler", region_name=REGION)
    return _scheduler


def get_sqs():
    global _sqs
    if _sqs is None:
        _sqs = boto3.client("sqs", region_name=REGION)
    return _sqs


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def submit_batch_job(queue: str, job_def: str, payload_dict: dict, job_name: str) -> str:
    """Submit a Batch job and return the job ID."""
    response = get_batch().submit_job(
        jobName=job_name,
        jobQueue=queue,
        jobDefinition=job_def,
        containerOverrides={
            "environment": [
                {"name": "JOB_PAYLOAD", "value": json.dumps(payload_dict)},
            ]
        },
    )
    job_id = response["jobId"]
    logger.info("Submitted Batch job %s to %s (job_def=%s)", job_id, queue, job_def)
    return job_id


def schedule_escalation(
    job_id: str,
    tier: int,
    payload: dict,
    receipt_handle: str,
    previous_job_id: str,
    delay_minutes: int = 30,
) -> None:
    """Create a one-time EventBridge Scheduler schedule to check/escalate this job."""
    schedule_name = f"whisper-escalate-{job_id}"
    fire_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
    # EventBridge Scheduler uses ISO 8601 without microseconds
    schedule_expression = f"at({fire_at.strftime('%Y-%m-%dT%H:%M:%S')})"

    escalation_payload = {
        "source": "escalation",
        "job_id": job_id,
        "tier": tier,
        "payload": payload,
        "receipt_handle": receipt_handle,
        "previous_job_id": previous_job_id,
    }

    get_scheduler().create_schedule(
        Name=schedule_name,
        ScheduleExpression=schedule_expression,
        ScheduleExpressionTimezone="UTC",
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": LAMBDA_ARN,
            "RoleArn": ESCALATION_SCHEDULE_ROLE_ARN,
            "Input": json.dumps(escalation_payload),
        },
        ActionAfterCompletion="DELETE",
    )
    logger.info(
        "Scheduled escalation check for job %s at %s (tier=%d)",
        job_id,
        fire_at.isoformat(),
        tier,
    )


def delete_schedule(job_id: str) -> None:
    """Delete the escalation schedule for a job (called on success)."""
    schedule_name = f"whisper-escalate-{job_id}"
    try:
        get_scheduler().delete_schedule(Name=schedule_name)
        logger.info("Deleted escalation schedule %s", schedule_name)
    except get_scheduler().exceptions.ResourceNotFoundException:
        logger.info("Schedule %s already gone, nothing to delete", schedule_name)
    except Exception as exc:
        logger.warning("Could not delete schedule %s: %s", schedule_name, exc)


def cancel_batch_job(job_id: str) -> None:
    """Cancel a stuck Batch job before escalating to the next tier."""
    try:
        get_batch().cancel_job(jobId=job_id, reason="Escalating to next tier")
        logger.info("Cancelled Batch job %s", job_id)
    except Exception as exc:
        logger.warning("Could not cancel job %s: %s", job_id, exc)


def delete_sqs_message(receipt_handle: str) -> None:
    """Delete the SQS message from the source queue."""
    if not SQS_QUEUE_URL:
        logger.warning("SQS_QUEUE_URL not set; cannot delete message")
        return
    get_sqs().delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
    logger.info("Deleted SQS message with receipt handle %s…", receipt_handle[:20])


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def handle_sqs_trigger(event: dict) -> dict:
    """
    Handle an SQS trigger event.

    Submits the job to Tier 1 (GPU spot) and schedules the first escalation
    check in 30 minutes. Returns batchItemFailures=[] so the Lambda SQS trigger
    does NOT auto-delete the message — we manage deletion ourselves on success.
    """
    record = event["Records"][0]
    receipt_handle = record["receiptHandle"]
    payload_dict = json.loads(record["body"])
    video_id = payload_dict.get("video_id", "unknown")

    tier = 1
    tier_info = TIER_MAP[tier]
    job_name = f"whisper-{video_id}-t{tier}"

    job_id = submit_batch_job(
        queue=tier_info["queue"],
        job_def=tier_info["job_def"],
        payload_dict=payload_dict,
        job_name=job_name,
    )
    logger.info(
        "Tier %d (%s) job submitted: %s for video_id=%s",
        tier,
        tier_info["label"],
        job_id,
        video_id,
    )

    schedule_escalation(
        job_id=job_id,
        tier=tier,
        payload=payload_dict,
        receipt_handle=receipt_handle,
        previous_job_id=job_id,
    )

    # Return empty batchItemFailures — we manage SQS deletion manually on success
    return {"batchItemFailures": []}


def handle_escalation(event: dict) -> dict:
    """
    Handle an EventBridge Scheduler escalation event.

    Checks the current Batch job status and either:
    - Succeeds: deletes SQS message and returns
    - Fails (and < tier 4): escalates to next tier immediately
    - Fails (tier 4): logs final failure, deletes SQS message (no more retries)
    - Still running (RUNNABLE/STARTING/PENDING and < tier 4): cancels job,
      escalates to next tier, schedules next check
    - Tier 4 still running: reschedules check (give CPU on-demand unlimited time)
    """
    job_id: str = event["job_id"]
    tier: int = int(event["tier"])
    payload: dict = event["payload"]
    receipt_handle: str = event["receipt_handle"]
    previous_job_id: str = event.get("previous_job_id", job_id)

    response = get_batch().describe_jobs(jobs=[previous_job_id])
    jobs = response.get("jobs", [])
    status = jobs[0]["status"] if jobs else "UNKNOWN"

    logger.info(
        "Escalation check: previous_job_id=%s status=%s tier=%d",
        previous_job_id,
        status,
        tier,
    )

    if status == "SUCCEEDED":
        logger.info("Job %s succeeded. Deleting SQS message.", previous_job_id)
        delete_sqs_message(receipt_handle)
        delete_schedule(previous_job_id)
        return {"status": "succeeded"}

    if status == "FAILED":
        if tier >= 4:
            logger.error(
                "Job %s FAILED at final tier %d. No more retries. Deleting SQS message.",
                previous_job_id,
                tier,
            )
            delete_sqs_message(receipt_handle)
            return {"status": "final_failure"}

        # Escalate to next tier immediately (no need to wait 30 more minutes)
        next_tier = tier + 1
        tier_info = TIER_MAP[next_tier]
        video_id = payload.get("video_id", "unknown")
        job_name = f"whisper-{video_id}-t{next_tier}"

        logger.info(
            "Job %s FAILED at tier %d. Escalating to tier %d (%s).",
            previous_job_id,
            tier,
            next_tier,
            tier_info["label"],
        )
        new_job_id = submit_batch_job(
            queue=tier_info["queue"],
            job_def=tier_info["job_def"],
            payload_dict=payload,
            job_name=job_name,
        )
        schedule_escalation(
            job_id=new_job_id,
            tier=next_tier,
            payload=payload,
            receipt_handle=receipt_handle,
            previous_job_id=new_job_id,
        )
        return {"status": "escalated", "new_job_id": new_job_id, "tier": next_tier}

    # Job is still in a pending/running state
    stuck_statuses = {"RUNNABLE", "STARTING", "PENDING", "SUBMITTED"}
    if status in stuck_statuses or status == "RUNNING":
        if tier >= 4:
            # CPU on-demand — give it unlimited time, just reschedule the check
            logger.warning(
                "Job %s still %s at tier 4 (CPU on-demand). Rescheduling check.",
                previous_job_id,
                status,
            )
            schedule_escalation(
                job_id=previous_job_id,
                tier=tier,
                payload=payload,
                receipt_handle=receipt_handle,
                previous_job_id=previous_job_id,
            )
            return {"status": "waiting_tier4", "job_id": previous_job_id}

        # Cancel stuck job and escalate to next tier
        next_tier = tier + 1
        tier_info = TIER_MAP[next_tier]
        video_id = payload.get("video_id", "unknown")
        job_name = f"whisper-{video_id}-t{next_tier}"

        logger.info(
            "Job %s stuck (%s) at tier %d. Cancelling and escalating to tier %d (%s).",
            previous_job_id,
            status,
            tier,
            next_tier,
            tier_info["label"],
        )
        cancel_batch_job(previous_job_id)
        new_job_id = submit_batch_job(
            queue=tier_info["queue"],
            job_def=tier_info["job_def"],
            payload_dict=payload,
            job_name=job_name,
        )
        schedule_escalation(
            job_id=new_job_id,
            tier=next_tier,
            payload=payload,
            receipt_handle=receipt_handle,
            previous_job_id=new_job_id,
        )
        return {"status": "escalated", "new_job_id": new_job_id, "tier": next_tier}

    # Unexpected status (e.g., UNKNOWN, STARTING in a completed sense) — log and reschedule
    logger.warning(
        "Job %s has unexpected status %s at tier %d. Rescheduling check.",
        previous_job_id,
        status,
        tier,
    )
    schedule_escalation(
        job_id=previous_job_id,
        tier=tier,
        payload=payload,
        receipt_handle=receipt_handle,
        previous_job_id=previous_job_id,
    )
    return {"status": "rescheduled", "job_status": status}


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    """Route the event to the appropriate handler."""
    logger.info("Received event source=%s", event.get("source", "<sqs>"))

    if event.get("source") == "escalation":
        return handle_escalation(event)

    # Default: SQS trigger (Records present)
    if "Records" in event and event["Records"]:
        return handle_sqs_trigger(event)

    logger.error("Unrecognized event shape: %s", json.dumps(event)[:200])
    return {"status": "unrecognised_event"}
