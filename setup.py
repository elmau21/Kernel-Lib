from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kernel-ml-engine",
    version="0.1.0",
    author="Kernel Team",
    author_email="contact@kernel-ml.com",
    description="Motor de métodos de kernel para machine learning e ingeniería",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kernel-ml/kernel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=12.0.0"],
        "dev": ["pytest>=7.4.0", "black>=23.10.0", "flake8>=6.1.0"],
    },
)

