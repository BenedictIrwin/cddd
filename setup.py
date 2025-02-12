from setuptools import setup

setup(
    name='cddd',
    version='1.2.2',
    packages=['cddd', 'cddd.data'],
    package_data={'cddd': ['data/*', 'data/default_model']},
    include_package_data=True,
    url='',
    download_url='',
    license='MIT',
    author='',
    author_email='',
    description='continous and data-driven molecular descriptors (CDDD)',
    python_requires='>=3.6.1, <3.8',
    install_requires=[
        'scikit-learn',
        'pandas<=1.0.3',
        'requests',
        'appdirs'
      ],
    extras_require = {
        'cpu': [
            'tensorflow==1.10.0'
            ]
    },
    entry_points={
        'console_scripts': [
            'cddd = cddd.run_cddd:main_wrapper',
        ],
    },
)
