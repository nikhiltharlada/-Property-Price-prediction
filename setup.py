from setuptools import setup,find_packages
from typing import List
error_d="-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace('\n',"") for req in requirements]
        if error_d in requirements:
            requirements.remove(error_d)
    return requirements





setup(
    name='property price prediction',
    version='0.0.1',
    author='Nikhil',
    author_email='nikhiltharlada310@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)