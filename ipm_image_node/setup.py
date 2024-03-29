from setuptools import setup

package_name = 'ipm_image_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='florian',
    maintainer_email='git@flova.de',
    description='Inverse Perspective Mapping Node for Image or Mask Topics',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ipm = ipm_image_node.ipm:main'
        ],
    },
)
