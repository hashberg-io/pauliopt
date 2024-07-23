clean_setup:
	pyenv virtualenv-delete -f $(shell cat .python-version)
	pyenv virtualenv $(shell cat .python-base-version) $(shell cat .python-version)
	#make setup
setup:
	pip install -r requirements-dev.txt
run_tests:
	python -m unittest discover -s ./tests/ -p "test_*.py"
upgrade_dependencies:
	pip-compile --allow-unsafe --no-annotate requirements-dev.in --upgrade
	pip-compile --allow-unsafe --no-annotate requirements.in --upgrade
