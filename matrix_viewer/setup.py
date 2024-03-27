from setuptools import setup

setup(
    name='matrix_viewer',
    version='0.1',
    packages=['matrix_viewer'],
    entry_points={
        'console_scripts': [
            'matrix-viewer = matrix_viewer.matrix_viewer:main'
        ]
    },
    install_requires=[
        'numpy',
        'tk'
    ],
    author='Felipe Loose',
    author_email='felipe.loose@gmail.com',
    description='A package for visualizing complex matrices'
)
