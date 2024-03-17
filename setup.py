from setuptools import setup, find_packages

with open("./requirements.txt", mode="r") as file:
    lines = file.readlines()
    requirements = [line.strip() for line in lines]

setup(
    name="nura",
    version="0.1.0",
    author="Andrew Holmes",
    author_email="andrewholmes011002@gmail.com",
    url="https://github.com/Andrew011002/deepent",
    python_requires=">=3.6",
    packages=find_packages(exclude=["*venv*"]),
    install_requires=requirements,
)
