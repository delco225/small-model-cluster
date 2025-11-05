# GPU Cluster Fiction Generation

Scripts and compose configuration for a multi-profile (cpu/gpu) AI cluster with:
- API Gateway (FastAPI) story orchestration
- Ollama (Phi3 mini) on CPU profile (fallback / lightweight model)
- vLLM (Mistral 7B) on GPU profile (medium option)
- vLLM (Llama 3 8B Instruct, quantized) multi-GPU (new) for higher quality responses
- Ray head/worker (coordination placeholder / future distributed scheduling)
- Storage (static volume)

## Prerequisites
- Docker / Docker Desktop (WSL2 backend recommended on Windows)
- For GPU profile: Ubuntu host with NVIDIA driver + nvidia-container-toolkit installed
- Pull Ollama model: automatically pulls `phi3:mini` at startup

## Environment Variables
Adjust in compose or set before running:
- MODEL_PHI3_ENDPOINT (default http://ollama-amd:11434 within compose network)
- MODEL_MISTRAL_ENDPOINT (http://mistral-vllm:8000 when GPU profile active)
- MODEL_LLAMA_ENDPOINT (http://llama-vllm:8001 when GPU profile active)
- MISTRAL_MODEL (default mistral-7b-instruct-v0.2)
- LLAMA_MODEL (default meta-llama/Meta-Llama-3-8B-Instruct) — provide a quantized variant for 4x 4GB GPUs
- PHI3_MODEL (default phi3:mini)
- GEN_MAX_TOKENS (default 640 currently; tune for latency)
- GEN_TEMPERATURE (default 0.7)

Quantization note: On four GTX 1650 SUPER cards you must use a 4-bit or 8-bit quantized 8B model to fit. If you pull full FP16 weights they will not fit in aggregate VRAM. Use AWQ/GPTQ/FP8 variants or convert locally.

## Scripts
### PowerShell (Windows)
```powershell
# Start CPU profile
./cluster/scripts/manage.ps1 -Profile cpu -Action up
# Start GPU profile (on server with GPUs)
./cluster/scripts/manage.ps1 -Profile gpu -Action up
# Build only api-gateway
./cluster/scripts/manage.ps1 -Action build
# Logs
./cluster/scripts/manage.ps1 -Action logs
# Stop
./cluster/scripts/manage.ps1 -Action down
```

### Bash (Linux/Ubuntu)
```bash
chmod +x cluster/scripts/manage.sh
# CPU profile
./cluster/scripts/manage.sh -p cpu -a up
# GPU profile
./cluster/scripts/manage.sh -p gpu -a up
# All profiles
./cluster/scripts/manage.sh -p all -a up
# Pre-pull images (recommended before first 'up')
./cluster/scripts/manage.sh -a pull
# Logs
./cluster/scripts/manage.sh -a logs
# Down
./cluster/scripts/manage.sh -a down
```

## Endpoints
- Health: `GET /health` shows story count and recent requests
- Perf: `GET /perf/recent` last timing metrics
- Init story: `GET/POST /story/init?title=My+Title&initial_paragraph=Opening...&total_parts=10`
- Next part: `POST /story/next/{story_id}`
- List stories: `GET /stories`
- Get story: `GET /story/{story_id}`
- Delete story: `DELETE /story/{story_id}`
- Direct generate (Phi3): `GET /generate/phi3?prompt=Hello`
- Direct generate (Mistral GPU): `GET /generate/mistral?prompt=Hello` (GPU profile)
- Direct generate (Llama multi-GPU): `GET /generate/llama?prompt=Hello` (GPU profile)

## Performance Tips
- Lower `max_tokens` query parameter for quicker responses
- Use smaller/quantized models on CPU (Phi3 mini) and reserve Mistral for GPU
- Monitor with `docker stats` and `/perf/recent`

## Deploy to Ubuntu GPU Server (Multi-GPU)
1. Install NVIDIA driver + nvidia-container-toolkit
2. Verify `docker info | grep -i nvidia` shows runtime (if using older setups add `--gpus all` or runtime spec)
3. Clone repo
4. Start CPU services first (optional): `./cluster/scripts/manage.sh -p cpu -a up`
5. Start GPU profile (includes Mistral + Llama): `./cluster/scripts/manage.sh -p gpu -a up`
6. Confirm containers: `docker compose -f cluster/gpu-cluster.yml ps`
7. Health check: `curl http://localhost:8080/health`

Optional pre-pull (speeds up first launch): `./cluster/scripts/manage.sh -a pull`

To limit GPUs (e.g., reserve fewer for Llama): edit `llama-vllm` service tensor-parallel-size and device count.

## Troubleshooting
- `No model succeeded`: verify endpoints with `curl http://localhost:11434/api/version`, `curl http://localhost:8000/v1/models`, `curl http://localhost:8001/v1/models`
- Slow responses: lower `GEN_MAX_TOKENS`, prefer Phi3 for draft, use Mistral or Llama only for final polish
- Out-of-memory (vLLM errors): ensure quantized weights, reduce `--max-model-len`, lower `max_tokens`
- 404 on `/story/init`: rebuild gateway: `docker compose -f cluster/gpu-cluster.yml build api-gateway`

## Next Steps
- Add Prometheus metrics
- Add persistence (SQLite) for stories
- Add batch generation to complete all parts automatically
- Add adaptive model selection logic based on prompt length & latency budget
- Add GPU memory telemetry endpoint
