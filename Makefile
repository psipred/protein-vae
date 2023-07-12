IMAGE := pspipred/protein-vae

.PHONY: build
build:
	docker build -f Dockerfile -t $(IMAGE) .

.PHONY: bash
bash:
	docker run --rm -it -v "$(PWD)":/app $(IMAGE) bash
