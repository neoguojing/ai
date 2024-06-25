# 定义变量
IMAGE_NAME=guojingneo/rag-app
DOCKERFILE_PATH=Dockerfile
CONTAINER_NAME=rag-app-container
PORT=7860

# 获取 Git 提交 ID
COMMIT_ID := $(shell git rev-parse --short HEAD)

# 默认目标
.PHONY: all
all: build

# 构建 Docker 镜像
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(COMMIT_ID) -f $(DOCKERFILE_PATH) .

# 运行 Docker 容器
.PHONY: run
run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME):$(COMMIT_ID)

# 停止并删除容器
.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# 推送 Docker 镜像到注册表
.PHONY: push
push:
	docker push $(IMAGE_NAME):$(COMMIT_ID)

# 清理未使用的 Docker 镜像和容器
.PHONY: clean
clean:
	docker system prune -f

# 打包镜像并推送
.PHONY: package
package: build push

# 显示帮助信息
.PHONY: help
help:
	@echo "使用方法:"
	@echo "  make build      构建 Docker 镜像"
	@echo "  make run        运行 Docker 容器"
	@echo "  make stop       停止并删除容器"
	@echo "  make push       推送 Docker 镜像到注册表"
	@echo "  make clean      清理未使用的 Docker 镜像和容器"
	@echo "  make package    构建并推送 Docker 镜像"
	@echo "  make help       显示帮助信息"
