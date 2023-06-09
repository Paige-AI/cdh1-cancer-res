from pathlib import Path
from typing import List

from setuptools import find_packages, setup

SOURCE_ROOT_DIRECTORY = Path(__file__).parent.resolve()


def read_lines(filename: Path) -> List[str]:
    with filename.open('r') as f:
        return f.readlines()


setup(
    name='cdh1-cancer-dis',
    version='0.0.0a0',
    python_requires='>=3.8',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=read_lines(SOURCE_ROOT_DIRECTORY / 'requirements.txt'),
    extras_require={},
)
