"""Create or verify portable SHA-256 manifests, streaming large files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/manifests/inputs.json')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        state = json.loads(args.verify.read_text())
        for name, expected in state['files'].items():
            path = ROOT/name
            if ROOT not in path.resolve().parents: raise ValueError('Path outside project')
            if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError('Hash mismatch: ' + name)
        print('VERIFIED', len(state['files']), 'files'); return
    files = {}
    for folder in ('datasets','pretrained_transformers','checkpoints'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
    if not files: raise ValueError('No input files found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(algorithm='SHA-256', files=files), indent=2), encoding='utf-8')
    print('MANIFEST', len(files), 'files', args.output)


if __name__ == '__main__': main()
