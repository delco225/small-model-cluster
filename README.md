# GPU Cluster Fiction Generation

Scripts and compose configuration for a two-profile (cpu/gpu) AI cluster with:
- API Gateway (FastAPI) story orchestration
- Ollama (Phi3 mini) on CPU profile
- vLLM (Mistral) on GPU profile (enable on Ubuntu server with NVIDIA drivers)
- Ray head/worker (coordination placeholder)
- Storage (static volume)

## Prerequisites
- Docker / Docker Desktop (WSL2 backend recommended on Windows)
- For GPU profile: Ubuntu host with NVIDIA driver + nvidia-container-toolkit installed
- Pull Ollama model: automatically pulls `phi3:mini` at startup

## Environment Variables
Adjust in compose or set before running:
- MODEL_PHI3_ENDPOINT (default http://ollama-amd:11434 within compose network)
- MODEL_MISTRAL_ENDPOINT (http://mistral-vllm:8000 when GPU profile active)
- MISTRAL_MODEL (default mistral-7b-instruct-v0.2)
- PHI3_MODEL (default phi3:mini)
- GEN_MAX_TOKENS (default 512)
- GEN_TEMPERATURE (default 0.7)

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

## Performance Tips
- Lower `max_tokens` query parameter for quicker responses
- Use smaller/quantized models on CPU (Phi3 mini) and reserve Mistral for GPU
- Monitor with `docker stats` and `/perf/recent`

## Deploy to Ubuntu GPU Server
1. Install NVIDIA driver + nvidia-container-toolkit
2. Clone repo
3. Enable GPU profile: `./cluster/scripts/manage.sh -p gpu -a up`
4. (Optional) uncomment deploy device reservations in compose

## Troubleshooting
- `No model succeeded`: verify endpoints with `curl http://localhost:11434/api/version` and `curl http://localhost:8000/v1/models`
- Slow responses: reduce `GEN_MAX_TOKENS`, trim prior parts, check CPU usage via `docker stats`
- 404 on `/story/init`: ensure gateway rebuilt after code change

## Next Steps
- Add Prometheus metrics
- Add persistence (SQLite) for stories
- Add batch generation to complete all parts automatically
