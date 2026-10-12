"""Download the OpenCV Zoo YOLOX CPU detector with pinned SHA-256 verification."""
import argparse
import hashlib
import os
from pathlib import Path
import requests

MODEL_NAME = 'object_detection_yolox_2022nov.onnx'
MODEL_SHA = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
MODEL_SIZE = 35858002
MODEL_URL = 'https://huggingface.co/opencv/opencv_zoo/resolve/main/models/object_detection_yolox/' + MODEL_NAME


def prepare(root, proxy=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / MODEL_NAME
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA:
        return path
    temp = path.with_suffix('.part')
    session = requests.Session()
    if proxy: session.proxies.update({'http': proxy, 'https': proxy})
    try:
        with session.get(MODEL_URL, stream=True, timeout=(15, 45)) as response:
            response.raise_for_status()
            digest, size = hashlib.sha256(), 0
            with temp.open('wb') as out:
                for chunk in response.iter_content(1024*1024):
                    size += len(chunk)
                    if size > MODEL_SIZE: raise ValueError('模型下载超过预期大小')
                    digest.update(chunk); out.write(chunk)
            if size != MODEL_SIZE or digest.hexdigest() != MODEL_SHA: raise ValueError('模型散列或大小校验失败')
        os.replace(temp, path)
        return path
    finally: temp.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy')
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1]/'data/person_models')
    args = parser.parse_args()
    print('Detector ready:', prepare(args.directory, args.proxy))
