import argparse
import json
import math
import yaml
import pathlib
import io
import threading
import queue
import subprocess

import boto3

from train import train, select_device

s3 = boto3.client("s3")

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data Ingest Args 
    parser.add_argument(
        "--manifest-bucket",
        type=str,
        help="input manifest/ label bucket name",
        default="network-machine-learning-data",
    )
    parser.add_argument(
        "--manifest-key", 
        type=str, 
        help="S3 object key for input manifest"
    )
    parser.add_argument(
        "--image-bucket",
        type=str,
        help="input image bucket (if not same as --bucket)",
        default="network-sandbox-processed-data",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.15,
        help="proportion of the input dataset to withold from training, for testing and validation (0.0 < n < 0.5)",
    )
    opt = parser.parse_args()
    if opt.image_bucket is None:
        opt.image_bucket = opt.bucket
    return opt


def main(opt):
    bucket = opt.bucket
    image_bucket = opt.image_bucket
    key = opt.key
    name = opt.name

    # download the desired dataset
    try:
        ingestion_process = subprocess.run([
            "python3", "download_ingested_dataset.py", 
            "--manifest-bucket", opt.manifest_bucket, 
            "--manifest-key", opt.manifest_key, 
            "--image-bucket", opt.image_bucket,
            "--holdout-size", opt.holdout_size,
        ], check=True)
    except subprocess.CalledProcessError as exc:
        print(ingestion_process.stdout)
        print(ingestion_process.stderr)
        raise exc

    # Shim in the yaml.data that we just generated on the fly from our Label Manifest from S3
    opt.data = yaml_path

    device = select_device(opt.device, batch_size=opt.batch_size)
    with open(opt.hyp, "r") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader) 

    # Delete duplicate argparse options
    # `Bucket` is used by both train.py (for some Google Cloud Storage we aren't accessing) 
    # and driver.py (for some S3 storage we are), so lets delete it before passing our args to train()
    del opt.bucket
    train(hyp=hyp, opt=opt, device=device, tb_writer=None)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)