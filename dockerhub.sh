#!/bin/bash
# Usage: ./dockerhub.sh <dockerhub-username> <repo-name>

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 noshore5 coherent-multiplex"
  exit 1
fi

USERNAME=$1
REPO=$2

# Get the current git commit hash
HASH=$(git rev-parse --short HEAD)

echo "Building Docker image: $USERNAME/$REPO:$HASH"

# Build the Docker image with the git hash as the tag
docker build -t $USERNAME/$REPO:$HASH .

if [ $? -ne 0 ]; then
    echo "Docker build failed!"
    exit 1
fi

# Log in to Docker Hub
echo "Logging into Docker Hub..."
docker login

if [ $? -ne 0 ]; then
    echo "Docker login failed!"
    exit 1
fi

# Push the image to Docker Hub
echo "Pushing image to Docker Hub..."
docker push $USERNAME/$REPO:$HASH

if [ $? -ne 0 ]; then
    echo "Docker push failed!"
    exit 1
fi

echo "Image $USERNAME/$REPO:$HASH uploaded to Docker Hub successfully."
