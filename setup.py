from setuptools import setup

setup(
    name='becca_viz',
    version='0.10.0',
    description='Visualization tools for Becca',
    url='http://github.com/brohrer/becca_viz',
    download_url='https://github.com/brohrer/becca_viz/archive/master.zip',
    author='Brandon Rohrer',
    author_email='brohrer@gmail.com',
    license='MIT',
    packages=['becca_viz'],
    include_package_data=True,
    install_requires=[
        'matplotlib',
    ],
    zip_safe=False)
