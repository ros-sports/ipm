import glob
import os

from setuptools import find_packages
from setuptools import setup

package_name = 'ipm_service'

setup(
    name=package_name,
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob.glob('launch/*.launch')),
    ],
    install_requires=[
        'launch',
        'setuptools',
    ],
    zip_safe=True,
    keywords=['ROS'],
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'ipm_service = ipm_service.ipm:main',
        ],
    }
)
