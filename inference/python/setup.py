import setuptools

VERSION = '0.0.1'

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

# with open("README.md", "r") as fh:
#     long_description = fh.read()
#
setuptools.setup(
    name="infbench",
    version=VERSION,
    author="Nathan Pemberton",
    author_email="nathanp@berkeley.edu",
    description="Support code for KaaS inference benchmark. This is not generally useful and shouldn't be used for anything else.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/nathantp/kaasBenchmarks.git",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.8',
    include_package_data=True
)
