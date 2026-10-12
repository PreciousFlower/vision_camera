"""Download the two public Kokoro CPU assets once; never runs a GPU job."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'data/speech_models'
BASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'
FILES = {'kokoro-v1.0.int8.onnx': 92361271, 'voices-v1.0.bin': 28214398}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy', help='Optional HTTP proxy for downloading public model files')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': args.proxy, 'https': args.proxy} if args.proxy else {}))
    def download(item):
        name, size = item
        target = ROOT / name
        if target.exists() and target.stat().st_size == size:
            return name, hashlib.sha256(target.read_bytes()).hexdigest()
        temporary = target.with_suffix(target.suffix + '.part')
        print('Downloading', name, flush=True)
        digest, received = hashlib.sha256(), 0
        try:
            with opener.open(BASE + name, timeout=30) as response, temporary.open('wb') as output:
                while chunk := response.read(1024*1024):
                    output.write(chunk); digest.update(chunk); received += len(chunk)
            if received != size:
                raise ValueError('Incomplete download: ' + name)
            temporary.replace(target)
            print('Ready', name, flush=True)
            return name, digest.hexdigest()
        finally:
            temporary.unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        hashes = dict(pool.map(download, FILES.items()))
    (ROOT / 'assets.json').write_text(json.dumps({'source': BASE, 'sha256': hashes}, indent=2), encoding='utf-8')
    print('Local natural voices ready. No GPU used.')


if __name__ == '__main__': main()
