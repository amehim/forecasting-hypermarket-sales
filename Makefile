setup:
	pip install -r requirements.txt

run:
	python src/train.py && src/holidays_us.py

forecast:
	jupyter notebook notebooks/02_modeling.ipynb

per-store:
	jupyter notebook notebooks/03_per_store_forecast.ipynb

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
