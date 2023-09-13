from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='floodprediction',
      version="0.0.1",
      description="Flood Prediction Model (api_pred)",
      license="LeWagon",
      author="Agustin Becker, Matias Duarte, Valentin Radovich",
      author_email="abecker@lewagon.org",
      url="https://github.com/Agubecker/flood_prediction",
      install_requires=requirements,
      packages=find_packages(),
      # test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
