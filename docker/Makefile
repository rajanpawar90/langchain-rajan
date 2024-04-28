# Makefile

IMAGE_NAME ?= graphdb
CONTAINER_NAME ?= graphdb

.PHONY: build start stop down

build:
	@echo "Building image ${IMAGE_NAME}"
	docker build --tag ${IMAGE_NAME} ./graphdb || { echo "Error building image, exiting..."; exit 1; }

start:
	@echo "Starting container ${CONTAINER_NAME}"
	docker-compose up -d ${CONTAINER_NAME} || { echo "Error starting container, exiting..."; exit 1; }

stop:
	@echo "Stopping container ${CONTAINER_NAME}"
	docker-compose down -v --remove-orphans || { echo "Error stopping container, exiting..."; exit 1; }

down:
	@echo "Removing container ${CONTAINER_NAME}"
	docker-compose down -v --remove-orphans || { echo "Error removing container, exiting..."; exit 1; }

.PHONY: clean

clean: down
	@echo "Removing image ${IMAGE_NAME}"
	docker rmi ${IMAGE_NAME} || { echo "Error removing image, exiting..."; exit 1; }

