docker build -t dash-prod-37-eg:1.0 -f Dockerfile.prod . 
docker run -d -p 80:80 dash-prod-37-eg:1.0 