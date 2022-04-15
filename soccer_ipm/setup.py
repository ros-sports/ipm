import glob
from setuptools import setup

package_name = 'soccer_ipm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/config",
            glob.glob('config/*.yaml')),
        ('share/' + package_name + '/launch',
            glob.glob('launch/*.launch')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='florian',
    maintainer_email='florian@flova.de',
    description='Inverse perspective mapping for the RoboCup soccer domain',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ipm = soccer_ipm.soccer_ipm:main',
        ],
    }
)
