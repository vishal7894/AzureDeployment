 docker buildx build --platform linux/amd64 --no-cache -t dash-prod-37-eg:1.0 -f Dockerfile.prod .
 docker image tag dash-prod-37-eg:1.0 pepsomest.azurecr.io/dash-37-eg:1.0
 docker push pepsomest.azurecr.io/dash-37-eg:1.0