from distutils.core import setup
from setuptools import find_packages

setup(
    name='msp-pupil-labs',
    version='1.0.8',
    author='Michael Barz',
    author_email='michael.barz@dfki.de',
    license='CC BY-NC-SA 4.0',
    packages=find_packages(
        include=('msp_pupil_labs.*', 'msp_pupil_labs')
    ),
    url="https://github.com/DFKI-Interactive-Machine-Learning/msp-pupil-labs",
    description="Pupil Labs Eye Tracking Extension for the DFKI multisensor pipeline framework.",
    python_requires='>=3.6.0',
    install_requires=[
        'multisensor-pipeline>=2.1.0',
        'pyzmq>=20.0.0',
        'Pillow>=8.3.2',
        'msgpack>1.0.0',
        'pupil-labs-realtime-api'
    ],
    keywords=[
        'eye tracking', 'multisensor-pipeline'
    ]
)
