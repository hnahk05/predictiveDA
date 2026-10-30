"""Sequential, artifact-verified queue. No training imports; Python 3.9 compatible."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def artifacts(folder, patterns):
    found = {}
    for pattern in patterns:
        if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
            raise ValueError('Artifact paths must stay inside attempt directory')
        matches = [p for p in folder.glob(pattern) if p.is_file() and p.stat().st_size]
        if not matches:
            raise ValueError('Missing/nonempty artifact: ' + pattern)
        for p in matches:
            if folder.resolve() not in p.resolve().parents:
                raise ValueError('Artifact escapes attempt directory')
            found[str(p.relative_to(folder))] = sha(p)
    return found


def fingerprint(job):
    h = hashlib.sha256(json.dumps(job, sort_keys=True).encode())
    h.update((sys.version + platform.system()).encode())
    for directory in ('models', 'runners', 'utils', 'configs', 'config_paper', 'scripts'):
        for p in sorted((ROOT / directory).rglob('*.py')):
            h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    h.update((ROOT / 'main.py').read_bytes())
    for name in ('environment/requirements-linux.txt', 'artifacts/environment/freeze-linux.txt'):
        path = ROOT / name
        if path.is_file(): h.update(path.read_bytes())
    return h.hexdigest()


def completed(folder, job, digest):
    for attempt in sorted(folder.glob('[0-9]*'), reverse=True):
        try:
            state = json.loads((attempt / 'result.json').read_text())
            if state['status'] == 'success' and state['fingerprint'] == digest:
                if artifacts(attempt, job['artifacts']) == state['artifact_sha256']:
                    return attempt
        except (OSError, ValueError, KeyError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--deadline', help='ISO time WITH offset, e.g. 2026-11-01T18:00:00+07:00')
    parser.add_argument('--buffer-minutes', type=float, default=20)
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline) if args.deadline else None
    if deadline and deadline.tzinfo is None:
        parser.error('Deadline must include timezone offset')
    if not math.isfinite(args.buffer_minutes) or args.buffer_minutes < 0:
        parser.error('Buffer must be finite and nonnegative')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    input_digest = ''
    if plan.get('input_manifest'):
        manifest_path = ROOT / plan['input_manifest']
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            path = ROOT / name
            if ROOT not in path.resolve().parents or sha(path) != expected['sha256']:
                parser.error('Input manifest mismatch: ' + name)
        input_digest = sha(manifest_path)
    jobs = plan['runs']
    seen = set()
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', job['id']) or job['id'] in seen:
            parser.error('Invalid/duplicate run ID')
        if not job.get('artifacts') or not job.get('command'):
            parser.error('Each job needs command and nonempty artifacts')
        if any(d not in seen for d in job.get('depends_on', [])):
            parser.error('Dependencies must precede job in plan')
        estimate = job.get('estimated_seconds', 0)
        if not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate < 0:
            parser.error('estimated_seconds must be finite and nonnegative')
        if deadline and estimate <= 0:
            parser.error('Deadline requires positive estimated_seconds for every job')
        seen.add(job['id'])
    args.runs = args.runs.resolve(); args.runs.mkdir(parents=True, exist_ok=True)
    lock = args.runs / '.queue.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error('Queue lock exists. Verify old process has stopped before removing ' + str(lock))
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    resolved, failed = {}, False
    try:
        for job in jobs:
            digest = hashlib.sha256((fingerprint(job) + input_digest).encode()).hexdigest()
            # Dependency attempt identities/hashes are part of the fingerprint.
            dependencies = job.get('depends_on', [])
            if any(d not in resolved for d in dependencies):
                print('BLOCKED', job['id'], flush=True); failed = True; continue
            if dependencies:
                digest = hashlib.sha256((digest + ''.join(
                    (resolved[d] / 'result.json').read_text() for d in dependencies)).encode()).hexdigest()
            folder = args.runs / job['id']; folder.mkdir(exist_ok=True)
            previous = completed(folder, job, digest)
            if previous:
                print('SKIP verified', previous, flush=True); resolved[job['id']] = previous; continue
            needed = job.get('estimated_seconds', 0) + args.buffer_minutes * 60
            if deadline and (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() <= needed:
                print('DEADLINE: no new run opened', job['id'], flush=True)
                return 3
            number = max([int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()] + [0]) + 1
            attempt = folder / f'{number:04d}'; attempt.mkdir()
            replacements = {'{python}': sys.executable, '{root}': str(ROOT), '{attempt}': str(attempt)}
            replacements.update({'{dependency:' + d + '}': str(resolved[d]) for d in dependencies})
            command = list(job['command'])
            for key, value in replacements.items(): command = [x.replace(key, value) for x in command]
            resumed_from = None
            if job.get('resume', False):
                for old in sorted(folder.glob('[0-9]*'), reverse=True):
                    if old == attempt: continue
                    try:
                        prior = json.loads((old/'result.json').read_text())
                        if prior['fingerprint'] == digest and prior['status'] != 'success' and list(old.rglob('*_[0-9]*.pt')):
                            resumed_from = str(old); command += ['--resume-dir', str(old)]; break
                    except (OSError, ValueError, KeyError): pass
                if deadline:
                    stop_at = deadline - dt.timedelta(minutes=args.buffer_minutes)
                    command += ['--stop-at', stop_at.isoformat()]
            state = dict(id=job['id'], attempt=number, command=command, fingerprint=digest, resumed_from=resumed_from,
                         started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='running')
            write(attempt / 'result.json', state)
            write(attempt / 'job.json', job)
            start = time.monotonic(); process = None
            try:
                print('RUN', job['id'], number, flush=True)
                with (attempt / 'console.log').open('wb') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state['exit_code'] = process.wait()
                if state['exit_code'] != 0: raise RuntimeError('Process returned nonzero')
                state['artifact_sha256'] = artifacts(attempt, job['artifacts'])
                state['status'] = 'success'; resolved[job['id']] = attempt
            except (Exception, KeyboardInterrupt) as error:
                if process is not None and process.poll() is None:
                    process.terminate(); process.wait()
                state['exit_code'] = process.returncode if process is not None else None
                state.update(status='failed', error=str(error)); failed = True
                if isinstance(error, KeyboardInterrupt): raise
            finally:
                state['elapsed_seconds'] = time.monotonic() - start
                state['ended_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
                write(attempt / 'result.json', state)
        return 1 if failed else 0
    finally:
        lock.unlink()


if __name__ == '__main__':
    sys.exit(main())
