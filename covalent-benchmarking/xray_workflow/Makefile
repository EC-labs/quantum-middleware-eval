venv:
	python3.8 -m venv .venv

activate: venv
	. .venv/bin/activate

install: activate
	pip install -r requirements.txt

run: install
	python script.py