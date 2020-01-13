from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='LGCN',
    version='1.0.0',
    description='Latent-Graph Convolutional Networks',
    author='Floris Hermsen',
    author_email='floris@owlin.com',
    packages=find_packages(),
    install_requires=requirements
)
