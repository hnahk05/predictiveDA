#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local_checks/tools artifacts/environment
# Keep the installer and all downloaded tools inside this project's ignored area.
curl --fail --location https://astral.sh/uv/install.sh --output .local_checks/tools/install_uv.sh
UV_INSTALL_DIR="$PWD/.local_checks/tools" UV_NO_MODIFY_PATH=1 sh .local_checks/tools/install_uv.sh
export UV_PYTHON_INSTALL_DIR="$PWD/.local_checks/python"
.local_checks/tools/uv python install 3.9
.local_checks/tools/uv venv --python 3.9 .local_checks/linux-env
.local_checks/tools/uv pip install --python .local_checks/linux-env/bin/python -r environment/requirements-linux.txt
.local_checks/tools/uv pip check --python .local_checks/linux-env/bin/python | tee artifacts/environment/pip-check.txt
.local_checks/tools/uv pip freeze --python .local_checks/linux-env/bin/python > artifacts/environment/freeze-linux.txt
.local_checks/linux-env/bin/python -m ensurepip
.local_checks/linux-env/bin/python -m pip check | tee artifacts/environment/python-pip-check.txt
.local_checks/linux-env/bin/python scripts/smoke_environment.py --output artifacts/environment/smoke-linux.json
