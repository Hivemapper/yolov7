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

    # Training Args
    parser.add_argument('--weights', type=str, default="''", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')

    return parser.parse_args()
    


def main(opt):
    # download the desired dataset
    ingestion_process = subprocess.run([
        "python3", "download_ingested_dataset.py", 
        "--manifest-bucket", str(opt.manifest_bucket), 
        "--manifest-key", str(opt.manifest_key), 
        "--image-bucket", str(opt.image_bucket),
        "--holdout-size", str(opt.holdout_size),
    ], check=True)

    train_command = [
        "python3", "train.py",
        "--name", str(opt.name),
        "--weights", str(opt.weights),
        "--cfg", str(opt.cfg),
        "--hyp", str(opt.hyp),
        "--epochs", str(opt.epochs),
        "--device", str(opt.device),
        "--workers", str(opt.workers),
        "--batch-size", str(opt.batch_size),
        "--data", "data/ingested-dataset.yaml",
    ]
    if opt.img_size:
        train_command.append("--img-size")
        for dimension_size in opt.img_size:
            train_command.append(str(dimension_size))
    if opt.evolve:
        train_command.append("--evolve")
    if opt.cache_images:
        train_command.append("--cache-images")
    if opt.multi_scale:
        train_command.append("--multi-scale")
    if opt.single_cls:
        train_command.append("--single-cls")
    if opt.adam:
        train_command.append("--adam")
    if opt.quad:
        train_command.append("--quad")
    if opt.linear_lr:
        train_command.append("--linear-lr")
    if opt.label_smoothing:
        train_command.extend(["--linear-lr", str(opt.label_smoothing)])
    if opt.save_period:
        train_command.extend(["--save-period", str(opt.save_period)])
    train_process = subprocess.run(train_command, check=True)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)