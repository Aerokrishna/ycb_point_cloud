import os
from glob import glob
from setuptools import setup

package_name = 'ycb_pointcloud_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krishnapranav',
    maintainer_email='you@example.com',
    description='RGB-D segmentation + point cloud extraction for YCB objects using RealSense D435i',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = ycb_pointcloud_pipeline.segmentation_node:main',
        ],
    },
)
