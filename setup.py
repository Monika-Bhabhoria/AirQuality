
from setuptools import find_packages,setup
from typing import List

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""

setup(
    name='AirQuality',
    version='0.0.1',
    author='Monika Bhabhoria',
    author_email='ms.monika5592@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)
"""
install_requires=get_requirements(requirement.txt) -- packages required for classes of local package to work,
"""