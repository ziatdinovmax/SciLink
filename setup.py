#!/usr/bin/env python3
"""
Setup script for SciLink package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="scilink",
    version="0.0.1",
    author="SciLink Team",
    author_email="maxim.ziatdinov@gmail.com",
    description="A framework for bridging experimental observations with literature analysis and computational materials modeling via large language models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/scilinkllm/scilink",
    project_urls={
        "Source": "https://github.com/scilink/scilink",
        "Tracker": "https://github.com/scilink/scilink/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "google-generativeai>=0.8.5",
        "atomai>=0.8.1",
        "futurehouse-client>=0.3.19",
        "ase>=0.8.5",
        "aimsgb>=1.1.1",
        "gdown>=5.2.0",
        "atomate2>=0.4.0",
    ],
    # Add CLI entry points
    entry_points={
        'console_scripts': [
            'scilink=scilink.cli.workflows:main',
        ],
    },
    include_package_data=True,
    package_data={
        "scilink": [
            "agents/sim_agents/docs/*.txt",
            "agents/sim_agents/docs/*.md",
        ],
    },
    zip_safe=False,
    keywords=[
        "materials science",
        "computational modeling",
        "machine learning",
        "large language models",
        "generative AI",
        "microscopy",
        "spectroscopy",
        "density functional theory",
        "atomic simulation",
        "experimental analysis",
        "literature mining"
    ],
)
