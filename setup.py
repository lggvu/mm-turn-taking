from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mm-turn-taking",
    version="0.1.0",
    description="Audio-VAP, Video-VAP and MM-VAP: multimodal Voice Activity Projection for turn-taking prediction",
    packages=find_packages(include=["mmvap", "mmvap.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)
