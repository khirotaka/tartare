# coding: utf-8
from setuptools import setup

requires = ["NumPy >=1.14.5",
            "SciPy >= 1.1.0",
            "Pillow >= 5.2.0",
            "Scikit-Learn >= 0.19.2"]

setup(
    name="tartare",
    version='0.1.1',
    description="Make homebrew image dataset for machine learning.",
    url="https://gitlab.com/Hiro2201/tartare",
    author="Hirotaka Kawashima",
    author_email="kawashima34@gmail.com",
    license='MIT',
    keywords="dataset",
    packages=[
        "tartare"
    ],
    install_requires=requires,
)
