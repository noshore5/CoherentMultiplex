# Usage: ./azure.sh <webapp-name> <resource-group> <dockerhub-username> <repo-name> <tag>

WEBAPP_NAME=$1
RESOURCE_GROUP=$2
DOCKERHUB_USER=$3
REPO=$4
TAG=$5

# Set the container image for the Azure Web App
az webapp config container set \
  --name $WEBAPP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $DOCKERHUB_USER/$REPO:$TAG \
  --docker-registry-server-url https://index.docker.io/v1/

# If your Docker Hub repo is private, add these lines:
# --docker-registry-server-user <dockerhub-username> \
# --docker-registry-server-password <dockerhub-password>

echo "Azure Web App $WEBAPP_NAME updated to use image $DOCKERHUB_USER/$REPO:$TAG from Docker Hub."
