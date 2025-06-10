from setuptools import find_packages,setup
from typing import List
e_dot="-e ."
def get_requirement(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if e_dot in requirements:
            requirements.remove(e_dot)
    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Nikhil',
    author_email='nikhiltharlada310@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirement.txt')
)