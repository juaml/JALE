from setuptools import find_packages, setup

setup(
    name="jale",  # Unique name for your package
    version="0.1.16",  # Version number
    description="Package allowing users to run Activation Likelihood Estimation Meta-Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lennart Frahm",
    author_email="l.frahm@mailbox.org",
    url="https://github.com/LenFrahm/JALE",
    license="MIT",
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=[  # List of dependencies
        "customtkinter>=5.2.2",
        "joblib>=1.3.2",
        "nibabel>=5.3.2",
        "numpy==1.26.3",
        "pandas>=2.2.3",
        "pytest>=8.0.0",
        "PyYAML>=6.0.2",
        "scipy>=1.14.1",
        "xgboost>=2.1.2",
        "openpyxl>=3.1.5",
        "scikit-learn-extra>=0.3.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "setuptools>=65.5.0",
    ],
    include_package_data=True,
    classifiers=[  # Classifiers help users find your project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
