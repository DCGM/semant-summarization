export now=$(date +"%Y-%m-%d %H:%M:%S")


docker compose -f ./docker-compose.yaml "$@"
