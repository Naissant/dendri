from setuptools import setup

setup(
    name="pysoma",
    version="0.0.1",
    description="Library for common healthcare related features.",
    author="PySoma Authors",
    license="MIT",
    packages=["pysoma"],
    install_requires=["pyspark>=3.0.0"],
    tests_require=["pytest"],
    classifiers=[
        "Development Status:: 2 - Pre-Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)
