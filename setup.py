from setuptools import find_packages, setup

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', errors='ignore') as ff:
    required = ff.read().splitlines()

setup(
    name='raidionicsseg',
    # package_dir={"": "raidionicsseg"},
    # packages=find_packages(where="raidionicsseg"),
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
    # include_package_data=True,
    python_requires=">=3.6",
    version='0.1.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='MIT',
    description='Raidionics segmentation and classification back-end with TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
