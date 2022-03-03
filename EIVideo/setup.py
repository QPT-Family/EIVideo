# Author: Acer Zhang
# Datetime: 2022/1/11
# Copyright belongs to the author.
# Please indicate the source for reprinting.
from setuptools import setup, find_packages

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

setup(
    name='EIVideo',
    version="0.1a2.dev4",
    packages=find_packages(),
    long_description="[https://github.com/QPT-Family/EIVideo](https://github.com/QPT-Family/EIVideo)",
    long_description_content_type='text/markdown',
    url='https://github.com/QPT-Family/EIVideo',
    license='LGPL',
    author='GT-ZhangAcer',
    author_email='zhangacer@foxmail.com',
    description='EIVideo - 交互式智能视频标注工具，几次鼠标点击即可解放双手，让视频标注更加轻松',
    install_requires=requirements,
    python_requires='>=3.7',
    include_package_data=True
)
