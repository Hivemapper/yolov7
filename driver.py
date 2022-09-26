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
    # Dataset Fetcher
    parser.add_argument("--bucket", help="input manifest/ label bucket name")
    parser.add_argument("--image-bucket", help="input image bucket (if not same as --bucket)")
    parser.add_argument("--key", help="S3 key")
    parser.add_argument(
        "--holdout-size", 
        type=float, 
        default=0.2, 
        help="proportion of the input dataset to hold out for testing and validation (0.0 < n < 1.0)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="MLFlow Experiment Name"
    )

    # Training Args
    parser.add_argument(
        "--weights", type=str, default="yolo7.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--hyp",
        type=str,
        default="data/hyp.scratch.p5.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="Upload dataset as W&B artifact table",
    )
    parser.add_argument(
        "--bbox_interval",
        type=int,
        default=-1,
        help="Set bounding-box image logging interval for W&B",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=-1,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument(
        "--artifact_alias",
        type=str,
        default="latest",
        help="version of dataset artifact to be used",
    )
    parser.add_argument(
        "--freeze",
        nargs="+",
        type=int,
        default=[0],
        help="Freeze layers: backbone of yolov7=50, first3=0 1 2",
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