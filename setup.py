import sys
from setuptools import find_packages, setup
import platform

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8', errors='ignore') as ff:
    required = ff.read().splitlines()

if platform.system() == 'Darwin' and platform.processor() == 'arm':  # Specific for Apple M1 chips
    required.append('onnxruntime-silicon')
else:
    required.append('onnxruntime')


setup(
    name='raidionicsseg',
    packages=find_packages(
        include=[
            'raidionicsseg',
            'raidionicsseg.Utils',
            'raidionicsseg.PreProcessing',
            'raidionicsseg.Inference',
            'tests',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsseg = raidionicsseg.__main__:main'
        ]
    },
    install_requires=required,
    python_requires=">=3.8",
    version='1.4.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics segmentation and classification back-end with ONNX runtime',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
