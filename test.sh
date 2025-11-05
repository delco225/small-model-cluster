#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# test.sh - Benchmark script to validate deployed models and save results
# This script tests external endpoints accessible from the host machine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${RESULTS_DIR}/test_result_${TIMESTAMP}.json"

# External endpoints (accessible from host)
API_GATEWAY="http://localhost:8080"
MISTRAL_DIRECT="http://localhost:8000"
LLAMA_DIRECT="http://localhost:8001"
OLLAMA_DIRECT="http://localhost:11434"
RAY_DASHBOARD="http://localhost:8265"

# Test prompts
SHORT_PROMPT="Write a brief story about a robot."
MEDIUM_PROMPT="Create a detailed story about an astronaut discovering alien life on Mars. Include dialogue and describe the alien creatures."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Initialize results file
init_results() {
    cat > "$RESULTS_FILE" << EOF
{
  "test_info": {
    "timestamp": "$(date -Iseconds)",
    "host": "$(hostname)",
    "script_version": "1.0"
  },
  "service_health": {},
  "model_tests": [],
  "performance_summary": {}
}
EOF
}

# Function to update JSON results
update_results() {
    local key="$1"
    local value="$2"
    
    python3 -c "
import json, sys
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['$key'] = $value
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || echo "Failed to update results for $key"
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    local health_results="{"
    
    # Check API Gateway
    log "Testing API Gateway ($API_GATEWAY/health)"
    local gateway_status=$(curl -s -w "HTTPSTATUS:%{http_code}" "$API_GATEWAY/health" 2>/dev/null || echo "HTTPSTATUS:000")
    local gateway_code=$(echo "$gateway_status" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    # Ensure gateway_code is valid
    if [[ ! "$gateway_code" =~ ^[0-9]+$ ]]; then
        gateway_code="0"
    fi
    
    if [ "$gateway_code" = "200" ]; then
        log "✓ API Gateway is healthy"
        health_results+='"api_gateway": {"status": "healthy", "code": 200},'
    else
        error "✗ API Gateway is not responding (HTTP $gateway_code)"
        health_results+='"api_gateway": {"status": "unhealthy", "code": '$gateway_code'},'
    fi
    
    # Check Ollama (Phi-3)
    log "Testing Ollama ($OLLAMA_DIRECT/api/version)"
    local ollama_status=$(curl -s -w "HTTPSTATUS:%{http_code}" "$OLLAMA_DIRECT/api/version" 2>/dev/null || echo "HTTPSTATUS:000")
    local ollama_code=$(echo "$ollama_status" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    # Ensure ollama_code is valid
    if [[ ! "$ollama_code" =~ ^[0-9]+$ ]]; then
        ollama_code="0"
    fi
    
    if [ "$ollama_code" = "200" ]; then
        log "✓ Ollama (Phi-3) is healthy"
        health_results+='"ollama": {"status": "healthy", "code": 200},'
    else
        warn "✗ Ollama is not responding (HTTP $ollama_code)"
        health_results+='"ollama": {"status": "unhealthy", "code": '$ollama_code'},'
    fi
    
    # Check Mistral vLLM
    log "Testing Mistral vLLM ($MISTRAL_DIRECT/v1/models)"
    local mistral_status=$(curl -s -w "HTTPSTATUS:%{http_code}" "$MISTRAL_DIRECT/v1/models" 2>/dev/null || echo "HTTPSTATUS:000")
    local mistral_code=$(echo "$mistral_status" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    # Ensure mistral_code is valid
    if [[ ! "$mistral_code" =~ ^[0-9]+$ ]]; then
        mistral_code="0"
    fi
    
    if [ "$mistral_code" = "200" ]; then
        log "✓ Mistral vLLM is healthy"
        health_results+='"mistral_vllm": {"status": "healthy", "code": 200},'
    else
        warn "✗ Mistral vLLM is not responding (HTTP $mistral_code)"
        health_results+='"mistral_vllm": {"status": "unhealthy", "code": '$mistral_code'},'
    fi
    
    # Check Llama vLLM
    log "Testing Llama vLLM ($LLAMA_DIRECT/v1/models)"
    local llama_status=$(curl -s -w "HTTPSTATUS:%{http_code}" "$LLAMA_DIRECT/v1/models" 2>/dev/null || echo "HTTPSTATUS:000")
    local llama_code=$(echo "$llama_status" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    # Ensure llama_code is valid
    if [[ ! "$llama_code" =~ ^[0-9]+$ ]]; then
        llama_code="0"
    fi
    
    if [ "$llama_code" = "200" ]; then
        log "✓ Llama vLLM is healthy"
        health_results+='"llama_vllm": {"status": "healthy", "code": 200},'
    else
        warn "✗ Llama vLLM is not responding (HTTP $llama_code)"
        health_results+='"llama_vllm": {"status": "unhealthy", "code": '$llama_code'},'
    fi
    
    # Check Ray Dashboard
    local ray_status=$(curl -s -w "HTTPSTATUS:%{http_code}" "$RAY_DASHBOARD" 2>/dev/null || echo "HTTPSTATUS:000")
    local ray_code=$(echo "$ray_status" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    # Ensure ray_code is valid
    if [[ ! "$ray_code" =~ ^[0-9]+$ ]]; then
        ray_code="0"
    fi
    
    if [ "$ray_code" = "200" ]; then
        log "✓ Ray Dashboard is healthy"
        health_results+='"ray_dashboard": {"status": "healthy", "code": 200}'
    else
        warn "✗ Ray Dashboard is not responding (HTTP $ray_code)"
        health_results+='"ray_dashboard": {"status": "unhealthy", "code": '$ray_code'}'
    fi
    
    health_results+="}"
    
    # Update results file
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['service_health'] = $health_results
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Test a model via API Gateway
test_model_via_gateway() {
    local model_name="$1"
    local prompt="$2"
    local max_tokens="${3:-256}"
    
    log "Testing $model_name via API Gateway..."
    
    # URL encode the prompt
    local encoded_prompt=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$prompt'''))")
    local url="$API_GATEWAY/generate/$model_name?prompt=$encoded_prompt&max_tokens=$max_tokens"
    
    local start_time=$(date +%s.%N)
    local response=$(curl -s -w "HTTPSTATUS:%{http_code};TIME:%{time_total}" "$url" 2>/dev/null || echo "ERROR")
    local end_time=$(date +%s.%N)
    
    if [[ "$response" == *"ERROR"* ]]; then
        error "Failed to connect to $model_name"
        return 1
    fi
    
    # Parse response
    local http_status=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local curl_time=$(echo "$response" | grep -o "TIME:[0-9.]*" | cut -d: -f2)
    local json_response=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*;//' | sed 's/TIME:[0-9.]*;//')
    
    if [ "$http_status" != "200" ]; then
        error "$model_name returned HTTP $http_status"
        echo "Response: $(echo "$json_response" | head -c 200)"
        return 1
    fi
    
    # Extract metrics
    local generated_text=$(echo "$json_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    text = data.get('text', '')
    print(text.replace('\n', ' ')[:150] + ('...' if len(text) > 150 else ''))
except Exception as e:
    print('JSON_PARSE_ERROR: ' + str(e))
" 2>/dev/null)
    
    local api_latency=$(echo "$json_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('latency_ms', 0))
except:
    print(0)
" 2>/dev/null)
    
    # Calculate approximate tokens
    local prompt_tokens=$(echo "$prompt" | wc -c | awk '{print int($1/4)}')
    local generated_tokens=$(echo "$generated_text" | wc -c | awk '{print int($1/4)}')
    
    # Calculate tokens per second
    local tokens_per_second=0
    if (( $(echo "$api_latency > 0" | bc -l 2>/dev/null || echo "0") )); then
        tokens_per_second=$(echo "scale=2; $generated_tokens / ($api_latency / 1000)" | bc 2>/dev/null || echo "0")
    fi
    
    log "✓ $model_name: ${generated_tokens} tokens in ${api_latency}ms (${tokens_per_second} tok/sec)"
    echo "  Generated: $generated_text"
    echo
    
    # Add result to JSON
    local test_result=$(cat << EOF
{
  "model": "$model_name",
  "prompt_tokens": $prompt_tokens,
  "generated_tokens": $generated_tokens,
  "max_tokens_requested": $max_tokens,
  "api_latency_ms": $api_latency,
  "curl_time_seconds": $curl_time,
  "tokens_per_second": $tokens_per_second,
  "http_status": $http_status,
  "generated_preview": "$(echo "$generated_text" | sed 's/"/\\"/g')",
  "success": true,
  "timestamp": "$(date -Iseconds)"
}
EOF
)
    
    # Append to results
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['model_tests'].append($test_result)
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Test direct model endpoints (bypass API Gateway)
test_direct_endpoints() {
    log "Testing direct model endpoints..."
    
    # Test Ollama directly
    log "Testing Ollama directly..."
    local ollama_payload='{"model": "phi3:mini", "prompt": "Write a short poem.", "options": {"num_predict": 100}}'
    local ollama_response=$(curl -s -X POST "$OLLAMA_DIRECT/api/generate" \
        -H "Content-Type: application/json" \
        -d "$ollama_payload" 2>/dev/null || echo "ERROR")
    
    if [[ "$ollama_response" != *"ERROR"* ]] && [[ "$ollama_response" == *"response"* ]]; then
        log "✓ Ollama direct endpoint working"
    else
        warn "✗ Ollama direct endpoint not working"
    fi
    
    # Test Ollama endpoints directly (all models now use Ollama)
    for service in "Mistral:$MISTRAL_DIRECT:mistral:7b-instruct" "Llama:$LLAMA_DIRECT:llama3.2:3b"; do
        local name=$(echo "$service" | cut -d: -f1)
        local endpoint=$(echo "$service" | cut -d: -f2-)
        local model=$(echo "$service" | cut -d: -f3-)
        
        log "Testing $name Ollama directly..."
        local ollama_payload='{"model": "'$model'", "prompt": "Hello", "options": {"num_predict": 10}}'
        local ollama_response=$(curl -s -X POST "$endpoint/api/generate" \
            -H "Content-Type: application/json" \
            -d "$ollama_payload" 2>/dev/null || echo "ERROR")
        
        if [[ "$ollama_response" != *"ERROR"* ]] && [[ "$ollama_response" == *"response"* ]]; then
            log "✓ $name Ollama direct endpoint working"
        else
            warn "✗ $name Ollama direct endpoint not working"
        fi
    done
}

# Generate performance summary
generate_summary() {
    log "Generating performance summary..."
    
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

tests = data.get('model_tests', [])
if not tests:
    print('No successful model tests found')
    exit(0)

print()
print('=' * 60)
print('MODEL PERFORMANCE SUMMARY')
print('=' * 60)

# Group by model
models = {}
for test in tests:
    model = test['model']
    if model not in models:
        models[model] = []
    models[model].append(test)

for model, model_tests in models.items():
    print(f'\\n{model.upper()}:')
    
    if model_tests:
        avg_latency = sum(t['api_latency_ms'] for t in model_tests) / len(model_tests)
        avg_tokens_per_sec = sum(t['tokens_per_second'] for t in model_tests if t['tokens_per_second'] > 0)
        if avg_tokens_per_sec > 0:
            avg_tokens_per_sec = avg_tokens_per_sec / len([t for t in model_tests if t['tokens_per_second'] > 0])
        
        total_tokens = sum(t['generated_tokens'] for t in model_tests)
        
        print(f'  Tests completed: {len(model_tests)}')
        print(f'  Average latency: {avg_latency:.1f}ms')
        print(f'  Average speed: {avg_tokens_per_sec:.1f} tokens/sec')
        print(f'  Total tokens generated: {total_tokens}')
        
        fastest = min(model_tests, key=lambda x: x['api_latency_ms'])
        print(f'  Fastest response: {fastest[\"api_latency_ms\"]}ms')

# Service health summary
print(f'\\nSERVICE HEALTH:')
health = data.get('service_health', {})
healthy_count = sum(1 for service, status in health.items() if status.get('status') == 'healthy')
total_count = len(health)
print(f'  Healthy services: {healthy_count}/{total_count}')

for service, status in health.items():
    status_icon = '✓' if status.get('status') == 'healthy' else '✗'
    print(f'  {status_icon} {service}: {status.get(\"status\", \"unknown\")}')

print()
print(f'Full results saved to: $RESULTS_FILE')
print('=' * 60)
"

    # Update summary in JSON
    python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

tests = data.get('model_tests', [])
health = data.get('service_health', {})

summary = {
    'total_tests': len(tests),
    'successful_tests': len([t for t in tests if t.get('success', False)]),
    'healthy_services': len([s for s in health.values() if s.get('status') == 'healthy']),
    'total_services': len(health)
}

if tests:
    summary['average_latency_ms'] = sum(t['api_latency_ms'] for t in tests) / len(tests)
    summary['total_tokens_generated'] = sum(t['generated_tokens'] for t in tests)

data['performance_summary'] = summary

with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Main execution
main() {
    log "Starting Model Validation and Benchmark Test"
    echo "Results will be saved to: $RESULTS_FILE"
    echo
    
    init_results
    check_service_health
    
    # Test models via API Gateway (this is the main functionality)
    local models=("mistral" "llama" "phi3")
    
    for model in "${models[@]}"; do
        if test_model_via_gateway "$model" "$SHORT_PROMPT" 256; then
            # If short test passes, try medium prompt
            test_model_via_gateway "$model" "$MEDIUM_PROMPT" 512
        fi
    done
    
    # Test direct endpoints for debugging
    test_direct_endpoints
    
    # Generate final summary
    generate_summary
    
    log "Test completed! Check results in $RESULTS_FILE"
}

# Check if running as script (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
# - Mistral 7B (vLLM on NVIDIA GPUs)
# - Llama 3 8B Instruct (vLLM multi-GPU on 4x NVIDIA GTX 1650 SUPER)
# - Phi-3 Mini (Ollama CPU/AMD fallback)
#
# Results are saved to test_result with JSON format for analysis.
#
# Usage:
#   ./test.sh [quick|full|stress]
#   quick  = Basic validation + single test per model (default)
#   full   = Multiple prompt sizes and parameter variations
#   stress = Load testing with concurrent requests

API_URL="http://localhost:8080"
TEST_MODE="${1:-quick}"
RESULTS_FILE="test_result"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEMP_FILE="/tmp/benchmark_$$"

# Test prompts of varying complexity
SHORT_PROMPT="Write a brief description of artificial intelligence."
MEDIUM_PROMPT="Create a detailed story about a robot learning emotions. Include character development and dialogue between the robot and its human companion."
LONG_PROMPT="Write an engaging science fiction story about humanity's first contact with an alien civilization. Include: 1) A diverse crew on a space station, 2) The discovery of mysterious signals, 3) First communication attempts, 4) Cultural misunderstandings, 5) A diplomatic breakthrough, and 6) The implications for Earth's future. Make it immersive with rich descriptions and character interactions."

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
    echo "[$(date +'%H:%M:%S')] $1" >> "$TEMP_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$TEMP_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $1" >> "$TEMP_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$TEMP_FILE"
}

# Initialize test results
initialize_results() {
    cat > "$TEMP_FILE" << EOF
GPU Model Benchmark Results
===========================
Timestamp: $(date -Iseconds)
Test Mode: $TEST_MODE
Host: $(hostname)
GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'nvidia-smi not available')

Test Results:
EOF
    
    log "Starting GPU Model Benchmark - Mode: $TEST_MODE"
    info "Results will be saved to: $RESULTS_FILE"
}

# Check cluster health and service availability
check_cluster_health() {
    log "Checking cluster health and service availability..."
    
    # Check API Gateway
    local gateway_response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$API_URL/health" 2>/dev/null || echo "HTTPSTATUS:000")
    local http_status=$(echo "$gateway_response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local json_body=$(echo "$gateway_response" | sed 's/HTTPSTATUS:[0-9]*$//')
    
    if [[ "$http_status" == "200" ]]; then
        info "✓ API Gateway is healthy: $json_body"
    else
        error "✗ API Gateway is not responding (HTTP: $http_status)"
        error "Make sure services are running: ./cluster/scripts/manage.sh -a up -p gpu"
        return 1
    fi
    
    # Check individual model endpoints
    local models=("mistral" "llama" "phi3")
    local available_models=()
    
    for model in "${models[@]}"; do
        log "Testing $model endpoint availability..."
        local test_response=$(curl -s -w "HTTPSTATUS:%{http_code}" "${API_URL}/generate/${model}?prompt=test&max_tokens=10" 2>/dev/null || echo "HTTPSTATUS:000")
        local status=$(echo "$test_response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        local body=$(echo "$test_response" | sed 's/HTTPSTATUS:[0-9]*$//')
        
        if [[ "$status" == "200" ]] && [[ "$body" != *'"error"'* ]]; then
            info "✓ $model endpoint is available"
            available_models+=("$model")
        else
            warn "✗ $model endpoint is not available (HTTP: $status)"
            echo "   Response: $(echo "$body" | head -c 100)..."
        fi
    done
    
    if [[ ${#available_models[@]} -eq 0 ]]; then
        error "No model endpoints are available. Cannot proceed with benchmarks."
        return 1
    fi
    
    info "Available models: ${available_models[*]}"
    echo "AVAILABLE_MODELS=${available_models[*]}" >> "$TEMP_FILE"
    return 0
}

# Check GPU utilization and memory
check_gpu_status() {
    log "Checking GPU status and utilization..."
    
    if command -v nvidia-smi &> /dev/null; then
        info "GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || warn "Could not query GPU status"
        echo
        
        # Save detailed GPU info
        echo "GPU_STATUS_BEFORE_TESTS:" >> "$TEMP_FILE"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null >> "$TEMP_FILE" || echo "GPU query failed" >> "$TEMP_FILE"
        echo "" >> "$TEMP_FILE"
    else
        warn "nvidia-smi not available - cannot check GPU utilization"
        echo "GPU_STATUS: nvidia-smi not available" >> "$TEMP_FILE"
    fi
}

# URL encode function for prompts
url_encode() {
    python3 -c "import urllib.parse; print(urllib.parse.quote('''$1'''))"
}

# Test individual model with specific parameters
test_model() {
    local model_name="$1"
    local prompt="$2"
    local prompt_type="$3"
    local max_tokens="${4:-512}"
    local temperature="${5:-0.7}"
    
    log "Testing $model_name with $prompt_type prompt (max_tokens=$max_tokens, temp=$temperature)"
    
    # Record start time
    local start_time=$(date +%s.%N)
    
    # Prepare URL and encode prompt
    local encoded_prompt=$(url_encode "$prompt")
    local url="${API_URL}/generate/${model_name}?prompt=${encoded_prompt}&max_tokens=${max_tokens}&temperature=${temperature}"
    
    # Make request with timing
    local response=$(curl -s -w "HTTPSTATUS:%{http_code};TIME:%{time_total}" "$url" 2>/dev/null || echo "ERROR;HTTPSTATUS:000;TIME:999")
    local end_time=$(date +%s.%N)
    
    # Parse response
    local http_status=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local curl_time=$(echo "$response" | grep -o "TIME:[0-9.]*" | cut -d: -f2)
    local json_response=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*;//' | sed 's/TIME:[0-9.]*;//')
    local wall_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
    
    # Validate response
    if [[ "$response" == *"ERROR"* ]] || [[ "$http_status" != "200" ]]; then
        error "$model_name test failed (HTTP: $http_status)"
        echo "Response: $(echo "$json_response" | head -c 200)..."
        
        # Log failure
        cat >> "$TEMP_FILE" << EOF
TEST_RESULT:
  Model: $model_name
  Prompt_Type: $prompt_type
  Status: FAILED
  HTTP_Status: $http_status
  Error: Connection or HTTP error
  Wall_Time: $wall_time
  Timestamp: $(date -Iseconds)

EOF
        return 1
    fi
    
    # Extract metrics from JSON response
    local metrics=$(python3 -c "
import json, sys
try:
    data = json.loads('''$json_response''')
    text = data.get('text', '')
    api_latency = data.get('latency_ms', 0)
    model = data.get('model', '$model_name')
    
    # Estimate tokens (rough: ~4 chars per token)
    prompt_chars = len('''$prompt''')
    generated_chars = len(text)
    prompt_tokens = max(1, prompt_chars // 4)
    generated_tokens = max(1, generated_chars // 4)
    
    # Calculate tokens per second
    tokens_per_sec = 0
    if api_latency > 0:
        tokens_per_sec = round(generated_tokens / (api_latency / 1000.0), 2)
    
    # Output metrics
    print(f'SUCCESS|{api_latency}|{generated_tokens}|{prompt_tokens}|{tokens_per_sec}|{text[:150].replace(chr(10), ' ')}')
except Exception as e:
    print(f'PARSE_ERROR|0|0|0|0|JSON parse failed: {str(e)[:50]}')
")
    
    # Parse metrics
    IFS='|' read -r status api_latency generated_tokens prompt_tokens tokens_per_sec generated_preview <<< "$metrics"
    
    if [[ "$status" != "SUCCESS" ]]; then
        error "$model_name response parsing failed: $generated_preview"
        
        cat >> "$TEMP_FILE" << EOF
TEST_RESULT:
  Model: $model_name
  Prompt_Type: $prompt_type
  Status: FAILED
  Error: JSON parsing failed
  Raw_Response: $(echo "$json_response" | head -c 100)
  Wall_Time: $wall_time
  Timestamp: $(date -Iseconds)

EOF
        return 1
    fi
    
    # Success - log results
    info "✓ $model_name completed successfully:"
    echo "   Generated: $generated_tokens tokens in ${api_latency}ms (${tokens_per_sec} tok/sec)"
    echo "   Preview: $generated_preview..."
    echo
    
    # Save detailed results
    cat >> "$TEMP_FILE" << EOF
TEST_RESULT:
  Model: $model_name
  Prompt_Type: $prompt_type
  Status: SUCCESS
  HTTP_Status: $http_status
  Prompt_Tokens: $prompt_tokens
  Generated_Tokens: $generated_tokens
  Max_Tokens_Requested: $max_tokens
  Temperature: $temperature
  API_Latency_Ms: $api_latency
  Wall_Time_Seconds: $wall_time
  Curl_Time_Seconds: $curl_time
  Tokens_Per_Second: $tokens_per_sec
  Generated_Preview: "$generated_preview"
  Timestamp: $(date -Iseconds)

EOF
    
    return 0
}

# Run concurrent stress test
stress_test_concurrent() {
    log "Running concurrent stress test (5 requests per available model)..."
    
    local models=($(grep "AVAILABLE_MODELS=" "$TEMP_FILE" | cut -d= -f2))
    local pids=()
    local stress_start=$(date +%s)
    
    # Launch concurrent requests
    for model in "${models[@]}"; do
        for i in {1..5}; do
            {
                local start=$(date +%s.%N)
                local result=$(curl -s "${API_URL}/generate/${model}?prompt=Write%20a%20creative%20short%20story%20about%20robots&max_tokens=200" 2>/dev/null || echo '{"error":"request_failed"}')
                local end=$(date +%s.%N)
                local duration=$(echo "$end - $start" | bc 2>/dev/null || echo "0")
                
                # Extract basic metrics
                local success="false"
                local tokens=0
                if [[ "$result" != *'"error"'* ]] && [[ "$result" == *'"text"'* ]]; then
                    success="true"
                    tokens=$(echo "$result" | python3 -c "import json,sys; data=json.load(sys.stdin); print(len(data.get('text',''))//4)" 2>/dev/null || echo "0")
                fi
                
                echo "STRESS_RESULT:$model:$i:$duration:$success:$tokens" >> "$TEMP_FILE"
            } &
            pids+=($!)
        done
    done
    
    # Wait for all requests
    local completed=0
    for pid in "${pids[@]}"; do
        if wait "$pid" 2>/dev/null; then
            ((completed++))
        fi
    done
    
    local stress_end=$(date +%s)
    local total_time=$((stress_end - stress_start))
    
    log "Stress test completed: $completed/${#pids[@]} requests in ${total_time}s"
    echo "STRESS_TEST_SUMMARY: $completed/${#pids[@]} requests in ${total_time}s" >> "$TEMP_FILE"
}

# Generate final benchmark summary
generate_summary() {
    log "Generating benchmark summary..."
    
    # Count results by model and status
    local summary=$(awk '
    BEGIN { 
        total_tests = 0
        failed_tests = 0 
        success_tests = 0
    }
    /TEST_RESULT:/ { 
        getline; model = $2
        getline; prompt_type = $2  
        getline; status = $2
        getline
        
        total_tests++
        models[model]++
        
        if (status == "SUCCESS") {
            success_tests++
            success_models[model]++
        } else {
            failed_tests++
            failed_models[model]++
        }
    }
    END {
        print "SUMMARY:"
        print "Total Tests: " total_tests
        print "Successful: " success_tests
        print "Failed: " failed_tests
        print ""
        print "Results by Model:"
        for (m in models) {
            succ = (m in success_models) ? success_models[m] : 0
            fail = (m in failed_models) ? failed_models[m] : 0
            print "  " m ": " succ " success, " fail " failed"
        }
    }' "$TEMP_FILE")
    
    echo
    info "$summary"
    echo "$summary" >> "$TEMP_FILE"
    
    # Check for stress test results
    local stress_results=$(grep "STRESS_RESULT:" "$TEMP_FILE" | wc -l)
    if [[ $stress_results -gt 0 ]]; then
        local stress_summary=$(awk -F: '
        /STRESS_RESULT:/ {
            model = $2
            success = ($5 == "true") ? 1 : 0
            duration = $4
            
            stress_total[model]++
            if (success) stress_success[model]++
            stress_time[model] += duration
        }
        END {
            print "Stress Test Results:"
            for (m in stress_total) {
                succ = (m in stress_success) ? stress_success[m] : 0
                avg_time = stress_time[m] / stress_total[m]
                print "  " m ": " succ "/" stress_total[m] " success, avg " avg_time "s"
            }
        }' "$TEMP_FILE")
        
        info "$stress_summary"
        echo "$stress_summary" >> "$TEMP_FILE"
    fi
}

# Main execution function
main() {
    initialize_results
    
    # Pre-flight checks
    if ! check_cluster_health; then
        error "Cluster health check failed. Exiting."
        exit 1
    fi
    
    check_gpu_status
    
    # Get available models from health check
    local available_models=($(grep "AVAILABLE_MODELS=" "$TEMP_FILE" | cut -d= -f2))
    
    if [[ ${#available_models[@]} -eq 0 ]]; then
        error "No models available for testing"
        exit 1
    fi
    
    # Run tests based on mode
    case "$TEST_MODE" in
        "quick")
            log "Running quick benchmark (single test per model)..."
            for model in "${available_models[@]}"; do
                test_model "$model" "$SHORT_PROMPT" "quick_test" 256 0.7 || warn "Quick test failed for $model"
            done
            ;;
            
        "full")
            log "Running full benchmark (multiple prompt sizes and parameters)..."
            for model in "${available_models[@]}"; do
                info "Testing $model with various configurations..."
                
                # Different prompt sizes
                test_model "$model" "$SHORT_PROMPT" "short" 256 0.7 || warn "Short test failed for $model"
                test_model "$model" "$MEDIUM_PROMPT" "medium" 512 0.7 || warn "Medium test failed for $model"
                test_model "$model" "$LONG_PROMPT" "long" 768 0.7 || warn "Long test failed for $model"
                
                # Different temperatures
                test_model "$model" "$MEDIUM_PROMPT" "creative" 512 0.9 || warn "Creative test failed for $model"
                test_model "$model" "$MEDIUM_PROMPT" "conservative" 512 0.3 || warn "Conservative test failed for $model"
            done
            ;;
            
        "stress")
            log "Running stress benchmark (concurrent load testing)..."
            
            # Pre-stress baseline
            for model in "${available_models[@]}"; do
                test_model "$model" "$SHORT_PROMPT" "pre_stress" 256 0.7 || warn "Pre-stress test failed for $model"
            done
            
            # Concurrent stress test
            stress_test_concurrent
            
            # Post-stress verification
            sleep 5
            check_gpu_status
            for model in "${available_models[@]}"; do
                test_model "$model" "$SHORT_PROMPT" "post_stress" 256 0.7 || warn "Post-stress test failed for $model"
            done
            ;;
            
        *)
            error "Unknown test mode: $TEST_MODE"
            echo "Supported modes: quick, full, stress"
            exit 1
            ;;
    esac
    
    # Post-test GPU status
    echo "GPU_STATUS_AFTER_TESTS:" >> "$TEMP_FILE"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null >> "$TEMP_FILE" || echo "GPU query failed" >> "$TEMP_FILE"
    fi
    echo "" >> "$TEMP_FILE"
    
    # Generate summary
    generate_summary
    
    # Move results to final file
    mv "$TEMP_FILE" "$RESULTS_FILE"
    
    log "Benchmark completed successfully!"
    info "Detailed results saved to: $RESULTS_FILE"
    
    # Display quick summary
    echo
    echo -e "${CYAN}=== QUICK SUMMARY ===${NC}"
    tail -15 "$RESULTS_FILE"
}

# Run main function
main "$@"