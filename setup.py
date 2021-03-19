from setuptools import setup

import versioneer

dev_requires = ["pytest", "black", "flake8", "pre-commit"]
extras = {
    "dev": dev_requires,
}

setup(
    name="dendri",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Dendri Authors",
    author_email="wesr000@gmail.com",
    description="Library for common healthcare related algorithms.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GDITHealth/dendri",
    license="MIT",
    packages=["dendri"],
    install_requires=["pyspark>=3.1.1", "python-dateutil"],
    extras_require=extras,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
)
