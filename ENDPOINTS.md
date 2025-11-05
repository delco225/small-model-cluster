# Docker Network Endpoints Explanation

## Why Different Endpoints?

### Internal Docker Network (Container-to-Container)
When containers communicate **within** the Docker Compose network, they use:
- `http://ollama-amd:11434` - Ollama service
- `http://mistral-vllm:8000` - Mistral vLLM service  
- `http://llama-vllm:8001` - Llama vLLM service
- `http://ray-head:6379` - Ray head service

These are **Docker service names** that resolve within the internal network.

### External Access (Host-to-Container)
When testing from your **host machine** (outside Docker), you must use:
- `http://localhost:8080` - API Gateway (FastAPI)
- `http://localhost:11434` - Ollama direct
- `http://localhost:8000` - Mistral vLLM direct
- `http://localhost:8001` - Llama vLLM direct
- `http://localhost:8265` - Ray Dashboard

These use the **port mappings** defined in docker-compose.yml.

## Current Configuration is CORRECT

The `api-gateway` environment variables should use **internal Docker names** because:

```yaml
api-gateway:
  environment:
    - MODEL_PHI3_ENDPOINT=http://ollama-amd:11434     # ✓ CORRECT (internal)
    - MODEL_MISTRAL_ENDPOINT=http://mistral-vllm:8000 # ✓ CORRECT (internal)  
    - MODEL_LLAMA_ENDPOINT=http://llama-vllm:8001     # ✓ CORRECT (internal)
```

The API Gateway **runs inside Docker** and communicates with other containers using internal service names.

## Test Script Uses External Endpoints

The `test.sh` script runs on your **host machine** and must use:

```bash
API_GATEWAY="http://localhost:8080"      # ✓ External access
MISTRAL_DIRECT="http://localhost:8000"   # ✓ External access  
LLAMA_DIRECT="http://localhost:8001"     # ✓ External access
OLLAMA_DIRECT="http://localhost:11434"   # ✓ External access
```

## Port Mappings Reference

| Service | Internal (Docker) | External (Host) | Purpose |
|---------|------------------|-----------------|---------|
| API Gateway | N/A | `localhost:8080` | Main FastAPI interface |
| Ollama | `ollama-amd:11434` | `localhost:11434` | Phi-3 Mini model |
| Mistral vLLM | `mistral-vllm:8000` | `localhost:8000` | Mistral-7B model |
| Llama vLLM | `llama-vllm:8001` | `localhost:8001` | Llama-3-8B model |
| Ray Dashboard | `ray-head:8265` | `localhost:8265` | Ray cluster UI |
| Storage | `storage:80` | `localhost:8090` | File storage |

## Testing Flow

1. **External Test** → `curl http://localhost:8080/generate/llama`
2. **API Gateway** → `requests.post('http://llama-vllm:8001/v1/completions')`
3. **Llama Container** → Returns response
4. **API Gateway** → Returns formatted response
5. **External Test** → Receives final result

This architecture allows:
- **External clients** to use one unified API (`localhost:8080`)
- **Internal services** to communicate efficiently via Docker network
- **Direct debugging** access to individual services when needed