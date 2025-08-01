#!/bin/bash
# Usage: ./upload_to_dockerhub.sh <dockerhub-username> <repo-name>

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 noshore5 coherent-multiplex"
  exit 1
fi

USERNAME=$1

REPO=$2
# Get the current git commit hash
HASH=$(git rev-parse --short HEAD)

# Build the Docker image with the git hash as the tag
docker build -t $USERNAME/$REPO:$HASH .

# Log in to Docker Hub
docker login


# Push the image to Docker Hub
docker push $USERNAME/$REPO:$HASH

echo "Image $USERNAME/$REPO:$TAG uploaded to Docker Hub."
