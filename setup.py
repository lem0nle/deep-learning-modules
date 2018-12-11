import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="deep-learning-modules",
    version="0.0.1",
    author="Le Dai",
    author_email="daile96@qq.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/doggoesroof/deep-learning-modules",
    packages=['dl'],
    # packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
