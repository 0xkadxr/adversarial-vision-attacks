"""Setup script for adversarial-vision-attacks."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adversarial-vision-attacks",
    version="0.1.0",
    author="Abdelkader Benmeriem",
    author_email="",
    description=(
        "Tools for generating adversarial images that expose vulnerabilities "
        "in multimodal LLMs."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kadirou12333/adversarial-vision-attacks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "jupyter"],
    },
)
