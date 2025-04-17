from setuptools import setup, find_packages

setup(
    name="lmapi",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "aiohttp>=3.8.1",
        "tiktoken>=0.6.0",
    ],
)
