#!/usr/bin/env bash
set -euo pipefail

# manage.sh - helper for building and running the AI fiction cluster
#
# Examples:
#   ./manage.sh -a build                 # Build api-gateway
#   ./manage.sh -a up -p cpu             # Start CPU profile services
#   ./manage.sh -a up -p gpu             # Start GPU services (after drivers installed)
#   ./manage.sh -a up -p all             # Start both profiles
#   ./manage.sh -a status                # Show container status
#   ./manage.sh -a story-init "Title" "Opening paragraph"  # Initialize a story
#   ./manage.sh -a story-next <story_id> # Generate next part
#   ./manage.sh -a perf                  # Show recent performance metrics
#   ./manage.sh -a logs                  # Tail logs
#   ./manage.sh -a down                  # Stop everything
#
# Requires: docker compose plugin

PROFILE="cpu"
ACTION="up"
COMPOSE_FILE="cluster/gpu-cluster.yml"
API_URL="http://localhost:8080"
TAIL_LINES=150

usage(){
  cat <<USAGE
Usage: $0 [-p cpu|gpu|all] [-a action] [args]
Actions:
  build            Build api-gateway image
  up               Start services for selected profile (cpu/gpu/all)
  down             Stop all services
  restart          Recreate services for profile
  logs             Show last $TAIL_LINES lines of logs
  status           List running containers in this project
  story-init TITLE OPENING [TOTAL_PARTS] [MODEL]
                   Initialize a multi-part story (default total_parts=10)
  story-next STORY_ID [MODEL] [MAX_TOKENS]
                   Generate next part for a story
  perf             Show recent performance entries
  curl PATH        Raw GET against API gateway (PATH like /health)
  help             Show this help
USAGE
  exit 1
}

while getopts ':p:a:' opt; do
  case $opt in
    p) PROFILE="$OPTARG";;
    a) ACTION="$OPTARG";;
    *) usage;;
  esac
done
shift $((OPTIND-1))

build_api(){
  echo "[build] api-gateway image";
  docker compose -f "$COMPOSE_FILE" build api-gateway;
}

up_profile(){
  if [ "$PROFILE" = "all" ]; then
    docker compose -f "$COMPOSE_FILE" --profile cpu --profile gpu up -d --build
  else
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d --build
  fi
}

restart_profile(){
  echo "[restart] profile=$PROFILE";
  up_profile
}

logs_all(){
  docker compose -f "$COMPOSE_FILE" logs --tail="$TAIL_LINES"
}

down_all(){
  echo "[down]";
  docker compose -f "$COMPOSE_FILE" down
}

status_list(){
  docker compose -f "$COMPOSE_FILE" ps
}

story_init(){
  local title="$1"; local opening="$2"; local total="${3:-10}"; local model="${4:-}";
  local url="$API_URL/story/init?title=$(python -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$title")&initial_paragraph=$(python -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$opening")&total_parts=$total";
  if [ -n "$model" ]; then url+="&preferred_model=$model"; fi
  curl -s "$url" | jq '.' || curl -s "$url"
}

story_next(){
  local sid="$1"; local model="${2:-}"; local max_tokens="${3:-}";
  local url="$API_URL/story/next/$sid";
  if [ -n "$model" ]; then url+="?preferred_model=$model"; fi
  if [ -n "$max_tokens" ]; then sep=$( [[ "$url" == *"?"* ]] && echo "&" || echo "?" ); url+="${sep}max_tokens=$max_tokens"; fi
  curl -s -X POST "$url" | jq '.' || curl -s -X POST "$url"
}

perf_recent(){
  curl -s "$API_URL/perf/recent" | jq '.' || curl -s "$API_URL/perf/recent"
}

raw_curl(){
  local path="$1"; curl -s "$API_URL$path" | jq '.' || curl -s "$API_URL$path"
}

case "$ACTION" in
  build) build_api;;
  up) up_profile;;
  down) down_all;;
  restart) restart_profile;;
  logs) logs_all;;
  status) status_list;;
  story-init) [ $# -ge 2 ] || usage; story_init "$1" "$2" "${3:-}" "${4:-}";;
  story-next) [ $# -ge 1 ] || usage; story_next "$1" "${2:-}" "${3:-}";;
  perf) perf_recent;;
  curl) [ $# -ge 1 ] || usage; raw_curl "$1";;
  help) usage;;
  *) usage;;
esac
