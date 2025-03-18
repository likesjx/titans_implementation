from setuptools import setup, find_packages

setup(
    name="titans_implementation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
    ],
    description="Implementation of Google's Titans AI architecture",
    author="Titans Team",
    author_email="your.email@example.com",
) 