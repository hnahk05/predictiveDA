"""Build an allowlisted cloud bundle. Never includes Windows/Linux virtualenvs."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'artifacts/cloud'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'predictiveDA-cloud.tar'
    manifest_path = ROOT / 'artifacts/manifests/inputs.json'
    manifest = json.loads(manifest_path.read_text())
    paths = {ROOT / name for name in manifest['files']}
    for folder in ('models','runners','utils','configs','config_paper','scripts','environment'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    # scripts/ also contains ignored personal debugging files: only include maintained automation.
    allowed_scripts = {'run_experiments.py','run_training.py','hash_manifest.py','smoke_environment.py','prepare_cloud.py','benchmark_gpu.py'}
    paths = {p for p in paths if p.parent != ROOT/'scripts' or p.name in allowed_scripts}
    paths.update([ROOT/'main.py', manifest_path, ROOT/'artifacts/environment/freeze-linux.txt'])
    if (ROOT/'docs/MISSIONS_GPU.md').exists(): paths.add(ROOT/'docs/MISSIONS_GPU.md')
    temporary = archive.with_suffix('.tmp')
    with tarfile.open(temporary, 'w') as tar:
        for path in sorted(paths):
            if ROOT not in path.resolve().parents: raise ValueError('Input path outside project')
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    temporary.replace(archive)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    (output/'predictiveDA-cloud.tar.sha256').write_text(h.hexdigest()+'  predictiveDA-cloud.tar\n')
    print('Bundle:', archive)
    print('Bytes:', archive.stat().st_size)
    print('SHA256:', h.hexdigest())


if __name__ == '__main__': main()
