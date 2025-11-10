from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='SINATRA-Pro',
    version='1.0.0',
    author="Wai Shing Tang",
    author_email="wai_shing_tang@brown.edu",
    description='Python package for SINATRA Pro - Protein Structure Analysis using Topological Data Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/lcrawlab/SINATRA-Pro",
    project_urls={
        "Bug Tracker": "https://github.com/lcrawlab/SINATRA-Pro/issues",
        "Documentation": "https://github.com/lcrawlab/SINATRA-Pro",
        "Source Code": "https://github.com/lcrawlab/SINATRA-Pro",
    },
    license='MIT License',
    license_files='LICENSE',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'torch>=1.10.0',
        'mdanalysis>=2.0.0',
        'fast-histogram>=0.11',
        'joblib>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='topological data analysis, protein dynamics, bioinformatics, euler characteristic transform',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        'console_scripts': [
            'sinatra-pro=sinatra_pro.__main__:main',
        ],
    },
)
