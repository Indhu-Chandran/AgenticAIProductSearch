from setuptools import setup, find_packages

setup(
    name="product_catalog_query_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai==1.3.0",
        "python-dotenv==1.0.0",
        "streamlit==1.24.0",
        "numpy",
        "pandas",
    ],
    python_requires=">=3.9.0",
)
