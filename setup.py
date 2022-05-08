from setuptools import find_packages, setup

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16', errors='ignore') as ff:
    required = ff.read().splitlines()

setup(
    name='raidionicsseg',
    packages=find_packages(
        include=[
            'raidionicsseg',
            'raidionicsseg.Utils',
            'raidionicsseg.PreProcessing',
            'raidionicsseg.Inference',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsseg = raidionicsseg.__main__:main'
        ]
    },
    install_requires=required,
    python_requires=">=3.6",
    version='0.1.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics segmentation and classification back-end with TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
