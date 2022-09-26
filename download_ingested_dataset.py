import copy
import io
import json
import math
import pathlib
import queue
import random
import sys
import threading
from typing import Any, Dict, List, Union

import boto3
import yaml

s3 = boto3.client("s3")

DATASET_MANIFEST_BUCKET = "network-machine-learning-data"
IMAGE_BUCKET = "network-sandbox-processed-data"
DATASET_MANIFEST_KEY = sys.argv[1]
TRAIN_VAL_SIZE = (
    0.2  # % of dataset to leave for validation (fitting) and testing (metrics)
)
DATASET_RELATIVE_PATH = pathlib.Path("./street-signs-from-ingest")


def main():
    manifest_content = get_manifest_content(DATASET_MANIFEST_KEY)
    manifest_items = load_jsonl(manifest_content)

    train_set, val_set, test_set = split_dataset(manifest_items)
    train_path, val_path, test_path = prep_dataset_directory(DATASET_RELATIVE_PATH)

    unpack_labels(train_set, train_path)
    download_images(train_set, train_path)

    unpack_labels(test_set, test_path)
    download_images(test_set, test_path)

    unpack_labels(val_set, val_path)
    download_images(val_set, val_path)

    write_dataset_yaml(manifest_items, pathlib.Path("data") / "dataset.yaml")


def load_jsonl(content: str):
    manifest_items = []
    for line in content.split("\n"):
        try:
            manifest_items.append(json.loads(line))
        except Exception as exc:
            print(f"failed to load line {line}, {exc}")
    return manifest_items


def get_manifest_content(manifest_key: str):
    with io.BytesIO() as buff:
        s3.download_fileobj(DATASET_MANIFEST_BUCKET, DATASET_MANIFEST_KEY, buff)
        buff.seek(0, 0)
        manifest_content = buff.read().decode("utf-8")
    return manifest_content


def prep_dataset_directory(dataset_root: pathlib.Path):
    test_dir = dataset_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    train_dir = dataset_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    val_dir = dataset_root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    for folder in (
        test_dir,
        train_dir,
        val_dir,
    ):
        img_dir = folder / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        label_dir = folder / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir, test_dir


def split_dataset(manifest_items: List[Any]):
    split_size = math.floor(len(manifest_items) * TRAIN_VAL_SIZE)
    train_set = copy.copy(manifest_items)
    random.shuffle(train_set)
    test_set = []
    val_set = []
    while len(test_set) < split_size:
        test_set.append(train_set.pop())
    while len(val_set) < split_size:
        val_set.append(train_set.pop())
    return train_set, val_set, test_set


def s3_key_to_local_path(s3Key: str):
    return s3Key.replace("/", "-")


def unpack_labels(manifest_items: List[Dict], data_subset_path: pathlib.Path):
    labels_dir = data_subset_path / "labels"
    for item in manifest_items:
        local_item_path = s3_key_to_local_path(item["s3Key"])
        label_file_path = (labels_dir / local_item_path).with_suffix(".txt")
        label_file_path.touch()
        format_label_file(label_file_path, item["labels"])
    return


def format_label_file(
    label_file_path: pathlib.Path, labels: List[Dict[str, Union[str, int, float]]]
):
    """
    Expect Label format: {
        bbox_cx: float,
        bbox_cy float,
        bbox_width: float,
        bbox_height: float,
        img_height: int,
        img_width: int,
        class: int,
        className: str,
    }
    """
    with label_file_path.open("w") as fp:
        for label in labels:
            class_id = label["class"]
            x_center_norm = label["bbox_cx"] / label["img_width"]
            box_width_norm = label["bbox_width"] / label["img_width"]
            y_center_norm = label["bbox_cy"] / label["img_height"]
            box_height_norm = label["bbox_height"] / label["img_height"]
            fp.write(
                f"{class_id} {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n"
            )
        return


def download_images(manifest_items: List[Dict], data_subset_path: pathlib.Path):
    images_dir = data_subset_path / "images"
    items_to_download = queue.Queue()

    def save_img(item):
        this_image_path = images_dir / s3_key_to_local_path(item["s3Key"])
        s3.download_file(IMAGE_BUCKET, item["s3Key"], str(this_image_path))
        return

    def save_img_worker():
        while True:
            unsaved_item = items_to_download.get()
            save_img(unsaved_item)
            items_to_download.task_done()

    for manifest_item in manifest_items:
        items_to_download.put(manifest_item)

    for i in range(50):
        threading.Thread(target=save_img_worker, daemon=True).start()
    items_to_download.join()


def write_dataset_yaml(manifest_items: List[Dict], write_path: pathlib.Path):
    output_yaml_data = {}
    class_labels_by_index = {}
    for item in manifest_items:
        for item_label in item["labels"]:
            class_labels_by_index[item_label["class"]] = item_label["className"]
    max_class_label_idx = max(class_labels_by_index.keys())
    class_labels = ["__UNREPRESENTED_CLASS__" for _ in range(max_class_label_idx)]
    for index, name in class_labels_by_index.items():
        class_labels[index - 1] = name
    output_yaml_data["names"] = class_labels
    output_yaml_data["nc"] = len(class_labels)
    output_yaml_data["train"] = str(DATASET_RELATIVE_PATH / "train")
    output_yaml_data["test"] = str(DATASET_RELATIVE_PATH / "test")
    output_yaml_data["val"] = str(DATASET_RELATIVE_PATH / "val")
    yaml_content = yaml.dump(output_yaml_data)
    with write_path.with_suffix(".yaml").open("w") as fp:
        fp.write(yaml.dump(output_yaml_data))
    return


if __name__ == "__main__":
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        main()
