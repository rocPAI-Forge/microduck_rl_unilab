.PHONY: sync sync-rocm sync-cuda

# Default profile: ROCm torch from pyproject.toml / uv.lock.
sync:
	uv sync

# Linux x86_64 with ROCm >= 7.1. Restores pyproject.rocm.toml and uv.rocm.lock.
sync-rocm:
	@cp pyproject.rocm.toml pyproject.toml
	@cp uv.rocm.lock uv.lock
	uv sync
	cp uv.lock uv.rocm.lock

# CUDA profile (cu128). Restores pyproject.cuda.toml and uv.cuda.lock.
# Switch back with: make sync-rocm
sync-cuda:
	@cp pyproject.cuda.toml pyproject.toml
	@cp uv.cuda.lock uv.lock
	uv sync
	cp uv.lock uv.cuda.lock
