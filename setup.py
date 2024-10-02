import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name="gallop",
    version="1.0.0",
    url="",
    packages=setuptools.find_packages(),
    author="Marc Lafon & Elias Ramzi",
    author_email="lafon.ma.ml@gmail.com",
    description="Adaptation of VLMs to downstream tasks using local and global prompts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)
