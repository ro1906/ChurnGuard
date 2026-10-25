.PHONY: install train run test lint docker

install:
	pip install -r requirements-dev.txt

train:
	python -m churnguard.train

run:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

docker:
	docker build -t churnguard . && docker run --rm -p 8501:8501 churnguard
