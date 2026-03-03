from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='fl_byzantine_library',
    version='0.1.0',
    description='A Federated Learning library with Byzantine-resilient aggregation and attack simulation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kerem Özfatura',
    author_email='aozfatura22@ku.edu.tr',
    url='https://github.com/CRYPTO-KU/FL-Byzantine-Library',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'fl-byzantine=main:main',
        ],
    },
)