from setuptools import setup, find_packages # type: ignore

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='language_checker',
    version='0.1.0',
    author='Łael Al-Halawani',
    author_email='laelhalawani@gmail.com',
    description='A helpful langauge detection AI using the meta\'s fasttext library and iso-639-3 language standard codes. \
Features language detection and conversion of language codes to human readable language names.',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9.0,<=3.10.14'
)