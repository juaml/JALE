from setuptools import find_packages, setup

setup(
    name="jale",  # Unique name for your package
    version="0.1.35",  # Version number
    description="Package allowing users to run Activation Likelihood Estimation Meta-Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lennart Frahm",
    author_email="l.frahm@mailbox.org",
    url="https://github.com/LenFrahm/JALE",
    license="MIT",
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=[  # List of dependencies
        "customtkinter==5.2.2",
        "joblib>=1.3.2",
        "nibabel>=5.2.1",
        "numpy==1.26.4",
        "pandas>=2.0.3",
        "PyYAML>=6.0.2",
        "scipy>=1.10.1",
        "xgboost>=2.1.2",
        "openpyxl>=3.1.5",
        "scikit-learn==1.5.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "nilearn>=0.10.4",
    ],
    include_package_data=True,
    classifiers=[  # Classifiers help users find your project
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.13",
)
