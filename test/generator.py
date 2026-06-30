"""
Script to blast images into S3.

This script runs locally and acts like 50 or more concurrent users all uploading images at the same time, 
to trigger the Lambda function and generate load. 

Supports both architectures:
  --arch monolith      → uploads to one InputBucket, encodes operation in key path
  --arch microservices → uploads to the operation-specific bucket (Resize/Greyscale/Detect)

Key format per architecture:
  monolith:       {run_id}/{operation}/{category}/{index:05d}_{filename}
  microservices:  {operation}/{run_id}/{category}/{index:05d}_{filename}

CloudFormation output keys expected:
  monolith:       InputBucketName
  microservices:  ResizeInputBucketName | GreyscaleInputBucketName | DetectInputBucketName
"""

import argparse 
import itertools
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed # for multithreading in order to simulate several users uploading images simulatneously
from datetime import datetime, timezone
import boto3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CATEGORIES = ("small", "medium", "large")
OPERATIONS = ("resize", "greyscale", "detect", "all")

# Maps operation name, thus CloudFormation output key for microservices buckets.
MICROSERVICES_OUTPUT_KEYS = {
    "resize":    "ResizeInputBucketName",
    "greyscale": "GreyscaleInputBucketName",
    "detect":    "DetectInputBucketName",
}

def resolve_bucket_name(bucket_arg, stack_name_arg, arch, operation):
    """Either use the bucket name given directly, or look it up from the
    CloudFormation stack's Outputs (the InputBucketName output defined in
    template.yaml)."""
    if bucket_arg:
        return bucket_arg

    if not stack_name_arg:
        sys.exit("Provide either --bucket <name> or --stack-name <name> (the stack you `sam deploy`ed).")

    if arch == "microservices" and operation == "all":
        sys.exit(
            "--operation all is only supported with --arch monolith.\n"
            "For microservices, choose: resize, greyscale, or detect."
        )

    target_output_key = MICROSERVICES_OUTPUT_KEYS[operation] if arch == "microservices" else "InputBucketName"

    cf = boto3.client("cloudformation")
    try:
        response = cf.describe_stacks(StackName=stack_name_arg)
    except Exception as exc:
        sys.exit(f"Could not find stack '{stack_name_arg}': {exc}")

    outputs = response["Stacks"][0].get("Outputs", [])
    for output in outputs:
        if output["OutputKey"] == target_output_key:
            return output["OutputValue"]

    sys.exit(f"Stack '{stack_name_arg}' has no output {target_output_key} - check the stack deployed correctly."
            f"({'microservices/template.yaml' if arch == 'microservices' else 'monolith/template.yaml'}).")

def load_dataset_pool(dataset_dir, category):
    """Returns a shuffled list of (category_label, filepath) tuples for the
    requested category, or for all three combined if category == 'mixed'."""
    wanted = CATEGORIES if category == "mixed" else (category,)

    pool = []
    for cat in wanted:
        folder = os.path.join(dataset_dir, cat)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                pool.append((cat, filepath))

    if not pool:
        sys.exit(
            f"No images found under {dataset_dir} for category '{category}'. "
            "Run test/dataset.py (download_and_split_coco_images) first."
        )

    random.shuffle(pool)
    return pool

def build_upload_plan(pool, count):
    """
    Instead of downloading 500 images, itertools.cycle creates an infinite loop 
    of your starting images count.
    """
    cycler = itertools.cycle(pool)
    return [next(cycler) for _ in range(count)]

def upload_one(s3_client, bucket, run_id, operation, arch, index, category, filepath):
    # creates a unique filename for S3
    filename = os.path.basename(filepath)

    if arch == "microservices":
        key = f"{operation}/{run_id}/{category}/{index:05d}_{filename}"
    else:
        key = f"{run_id}/{operation}/{category}/{index:05d}_{filename}"

    start = time.perf_counter()
    try:
        with open(filepath, "rb") as f:
            s3_client.put_object(Bucket=bucket, Key=key, Body=f.read())
        latency_ms = (time.perf_counter() - start) * 1000
        return {"key": key, "category": category, "ok": True, "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {"key": key, "category": category, "ok": False, "latency_ms": latency_ms, "error": str(exc)}


def run(args):
    bucket = resolve_bucket_name(args.bucket, args.stack_name, args.arch, args.operation)
    pool = load_dataset_pool(args.dataset_dir, args.category)
    plan = build_upload_plan(pool, args.count)

    print(f"Architecture: {args.arch}")
    print(f"Bucket:       {bucket}")
    print(f"Operation:    {args.operation}")
    print(f"Category:     {args.category}")
    print(f"Count:        {args.count}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Run ID:       {args.run_id}")
    print("Uploading...")

    s3_client = boto3.client("s3")
    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool_executor:
        futures = [
            pool_executor.submit(upload_one, s3_client, bucket, args.run_id, args.operation, args.arch, i, category, filepath)
            for i, (category, filepath) in enumerate(plan)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % max(1, len(futures) // 10) == 0 or completed == len(futures):
                print(f"  {completed}/{len(futures)} done")

    wall_time_s = time.perf_counter() - wall_start
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    latencies = [r["latency_ms"] for r in successes]

    summary = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arch": args.arch,
        "bucket": bucket,
        "operation": args.operation,
        "category": args.category,
        "requested_count": args.count,
        "concurrency": args.concurrency,
        "succeeded": len(successes),
        "failed": len(failures),
        "wall_time_s": round(wall_time_s, 3),
        "throughput_img_per_s": round(len(successes) / wall_time_s, 2) if wall_time_s > 0 else None,
        "upload_latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(statistics.quantiles(latencies, n=20)[18], 1) if len(latencies) >= 20 else None,
            "min": round(min(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "failures": failures,
    }

    print("\n" + "=" * 50)
    print(f"Done: {summary['succeeded']} succeeded, {summary['failed']} failed")
    print(f"Wall time: {summary['wall_time_s']}s  |  Throughput: {summary['throughput_img_per_s']} img/s")
    if summary["upload_latency_ms"]["mean"] is not None:
        print(f"Upload latency (ms) - mean: {summary['upload_latency_ms']['mean']}, "
              f"median: {summary['upload_latency_ms']['median']}, "
              f"max: {summary['upload_latency_ms']['max']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{args.run_id}.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {result_path}")

    # client-side view of the experiment. Cross-reference it
    # against CloudWatch (Lambda Duration/Invocations/ConcurrentExecutions,
    # and the custom ImagePipeline/ProcessingDurationMs metric from app.py)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload images into the pipeline's S3 input bucket.")
    bucket_group = parser.add_mutually_exclusive_group()
    bucket_group.add_argument("--bucket", help="Input bucket name (skips the CloudFormation lookup)")
    bucket_group.add_argument("--stack-name", help="SAM/CloudFormation stack name to look up InputBucketName from")
    parser.add_argument("--arch", choices=["monolith", "microservices"], default="monolith", help="Which architecture to use")
    parser.add_argument("--operation", choices=OPERATIONS, default="all", help="Which operation the Lambda should perform")
    parser.add_argument("--category", choices=(*CATEGORIES, "mixed"), default="mixed")
    parser.add_argument("--count", type=int, default=50, help="Total number of images to upload")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of parallel upload workers (simulated concurrent users)")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Path to the folder containing small/medium/large subfolders")
    parser.add_argument("--run-id", default=datetime.now().strftime("run-%Y%m%d-%H%M%S"))

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
