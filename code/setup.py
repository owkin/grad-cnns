from setuptools import setup, find_packages


setup(name='gradcnn',
        version='0',
        install_requires=[
            'numpy',
            'torch',
        ],
        description='Efficient Per-Example Gradient Computations in Convolutional Neural Networks',
        author='Gaspar Rochette, Andre Manoel, Eric Tramel',
        packages=find_packages(),
     )
