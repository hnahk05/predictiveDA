# Build on a Linux machine with Docker; not built on this Windows host.
FROM python:3.9-slim-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 openssh-server curl ca-certificates git procps && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/predictiveDA
COPY environment/requirements-linux.txt /tmp/requirements-linux.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-linux.txt && python -m pip check
COPY . .
# Save the actual resolved environment at image build time.
RUN mkdir -p artifacts/environment && python -m pip freeze > artifacts/environment/freeze-image.txt
CMD ["bash"]
