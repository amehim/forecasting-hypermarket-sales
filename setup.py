from setuptools import setup, find_packages

setup(
    name="hypermarket_holiday_forecast",
    version="1.0.0",
    description="Forecasting hypermarket holiday sales using Prophet",
    author="Idoga Ameh",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "prophet"
    ],
)
