# -*- codeing = utf-8 -*-
# @Time : 2024-05-26 15:28
# @Author : 张庭恺
# @File : setup.py
# @Software : PyCharm
from setuptools import setup, find_packages

setup(
    name='pointnet2',
    version='0.1',
    packages=find_packages(include=['Pointnet_Pointnet2','Pointnet_Pointnet2.*']),
    install_requires=[
        # 在这里列出你的包依赖，例如：
        # 'requests',
    ],
    author='Point',
    author_email='861482002@qq.com',
    description='点云特征提取',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='你的项目地址，例如 GitHub 仓库地址',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
