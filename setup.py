import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='image_augmenter',
    version='0.2.4',
    author='Michele Costa',
    author_email='costajob@gmail.com',
    description='A tiny python library to augment the images dataset aimed for a ML classification system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/costajob/image_augmenter',
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib>=3.0',
        'numpy>=1.16',
        'Pillow>=5.4',
        'scikit-image>=0.14',
        'scipy>=1.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
