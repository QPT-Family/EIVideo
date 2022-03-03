# Author: Acer Zhang
# Datetime: 2022/1/14 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
from setuptools import setup, find_packages


with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

setup(
    name='QEIVideo',
    version="0.1a1.dev2",
    packages=find_packages(),
    long_description="[https://github.com/QPT-Family/EIVideo](https://github.com/QPT-Family/EIVideo)",
    long_description_content_type='text/markdown',
    url='https://github.com/QPT-Family/EIVideo',
    license='LGPL',
    author='GT-ZhangAcer',
    author_email='zhangacer@foxmail.com',
    description='QEIVideo - 交互式智能视频标注工具前端交互支持',
    install_requires=requirements,
    python_requires='>=3.7',
    include_package_data=True
)
