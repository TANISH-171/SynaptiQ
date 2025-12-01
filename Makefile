.PHONY: api ui install format test

install:
	python -m pip install --upgrade pip
	pip install -r backend/requirements.txt
	pip install -r frontend/requirements.txt
	@echo "Done. Activate venv before running."

api:
	uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload

ui:
	API_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501

format:
	python -m pip install black isort ruff
	black .
	isort .
	ruff check .

.PHONY: test
test:
	python -m pip install -r backend/requirements.txt || true
	set PYTHONPATH=backend
	pytest -q backend/tests

