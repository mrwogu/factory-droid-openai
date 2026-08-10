# syntax=docker/dockerfile:1.7
#
# Container image for the Factory Droid OpenAI Bridge.
#
# Builds a self-contained image that runs the bridge and a pinned Droid CLI.
# The bridge listens on 0.0.0.0:8787 inside the container; bind it to
# 127.0.0.1 on the host and front it with the bridge bearer token when
# exposing it beyond loopback.
#
# Build args:
#   PYTHON_VERSION   Python base image tag (default 3.13-slim).
#   DROID_VERSION    Factory Droid CLI version (default: latest from npm registry).
#   BRIDGE_VERSION   factory-droid-openai PyPI version (default: install from source).
#
# Droid auto-update is disabled so the resolved CLI version is locked in for the
# lifetime of the image. Omit DROID_VERSION to auto-resolve latest at build time;
# set it to pin a specific release.

ARG PYTHON_VERSION=3.13-slim
ARG DROID_VERSION

FROM python:${PYTHON_VERSION} AS base

ARG DROID_VERSION
ARG BRIDGE_VERSION

ENV FACTORY_DROID_OPENAI_HOST=0.0.0.0 \
    FACTORY_DROID_OPENAI_PORT=8787 \
    FACTORY_DROID_OPENAI_WORKDIR=/work \
    FACTORY_DROID_AUTO_UPDATE_ENABLED=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# Install the Droid CLI binary with SHA256 verification.
# When DROID_VERSION is unset, resolve the latest release from the npm
# registry first. Direct binary download is used instead of the curl|sh
# installer so the image is reproducible and the checksum is enforced.
RUN set -e; \
    _droid_version="${DROID_VERSION:-}"; \
    if [ -z "${_droid_version}" ]; then \
      _droid_version=$(curl -fsSL https://registry.npmjs.org/@factory/cli/latest \
        | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4); \
      if [ -z "${_droid_version}" ]; then \
        echo "Failed to resolve latest Droid CLI version from npm registry" >&2; \
        exit 1; \
      fi; \
      echo "Resolved latest Droid CLI: ${_droid_version}"; \
    fi; \
    case "$(uname -m)" in \
      x86_64) arch=x64 ;; \
      aarch64) arch=arm64 ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    base="https://downloads.factory.ai/factory-cli/releases/${_droid_version}/linux/${arch}"; \
    curl -fsSLO "${base}/droid"; \
    curl -fsSLO "${base}/droid.sha256"; \
    echo "$(awk '{print $1}' droid.sha256)  droid" | sha256sum --check -; \
    chmod +x droid; \
    mv droid /usr/local/bin/droid; \
    rm -f droid.sha256; \
    droid --version

# Install the bridge. When BRIDGE_VERSION is set, pull from PyPI; otherwise
# install from the copied source tree (used for development and CI builds).
# /work is created first so the bridge's Settings.from_env validation passes
# during the --help smoke check; it is reused as the runtime working directory.
RUN mkdir -p /work
WORKDIR /app
COPY . .
RUN if [ -n "${BRIDGE_VERSION}" ]; then \
      pip install --no-cache-dir "factory-droid-openai==${BRIDGE_VERSION}"; \
    else \
      pip install --no-cache-dir .; \
    fi; \
    factory-droid-openai --help >/dev/null

# Non-root runtime. The bridge and Droid subprocess never write to the
# working directory (Factory-native tools are disabled), so a read-only host
# mount at /work is safe. Sessions live in the per-user Factory config dir.
RUN useradd --create-home --uid 1000 droid \
 && mkdir -p /home/droid/.factory \
 && chown -R droid:droid /work /home/droid/.factory
USER droid
WORKDIR /work

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8787/health || exit 1

CMD ["factory-droid-openai"]
