from setuptools import setup, find_packages

setup(
    name='gym_anytrading',
    version='1.3.2',
    packages=find_packages(),

    author='AminHP',
    author_email='mdan.hagh@gmail.com',
    license='MIT',

    install_requires=[
        'gymnasium>=0.28',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1'
    ],

    package_data={
        'gym_anytrading': ['datasets/data/*']
    }
)
