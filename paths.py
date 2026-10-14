"""统一的路径设置。

RAW_DATA：期货数据目录，由环境变量 CTA_RAW_DATA 指定，例如
          Windows（PowerShell）：[Environment]::SetEnvironmentVariable('CTA_RAW_DATA', 'D:\\data\\raw_data', 'User')
          Linux / macOS：export CTA_RAW_DATA=/data/raw_data
          未设置时退回到本仓库旁边的 raw_data/（<上一级目录>/raw_data），并给出提示。
RESEARCH：本仓库（research/）的根目录。
"""
import os
import sys

RESEARCH = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')

_env = os.environ.get('CTA_RAW_DATA')
if _env:
    RAW_DATA = _env.replace(os.sep, '/')
else:
    RAW_DATA = os.path.join(os.path.dirname(RESEARCH), 'raw_data').replace(os.sep, '/')
    print(f'[paths] 未设置环境变量 CTA_RAW_DATA，使用默认数据目录 {RAW_DATA}', file=sys.stderr)

if not os.path.isdir(RAW_DATA):
    raise FileNotFoundError(f'数据目录不存在：{RAW_DATA}。请设置环境变量 CTA_RAW_DATA 指向 raw_data 文件夹。')
