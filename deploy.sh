#!/bin/bash

# Nome da imagem
IMAGE_NAME="langchain-agent-app"

echo "📦 Construindo imagem Docker..."
docker build -t $IMAGE_NAME:latest .

echo "🚀 Subindo para o Minikube..."
eval $(minikube docker-env)
docker build -t $IMAGE_NAME:latest .

echo "📌 Aplicando Deploy no Minikube..."
kubectl apply -f deployment.yaml

echo "🔄 Atualizando Pods..."
kubectl rollout restart deployment langchain-agent-app

echo "✅ Deploy atualizado com sucesso!"
