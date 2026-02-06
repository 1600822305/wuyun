"""
丘脑模块 (Thalamus)

提供丘脑核团和路由器:
- ThalamicNucleus: 单个丘脑核团 (TC + TRN)
- create_thalamic_nucleus: 核团工厂函数
- ThalamicRouter: 多核团路由器 + TRN 竞争抑制
"""

from wuyun.thalamus.thalamic_nucleus import (
    ThalamicNucleus,
    create_thalamic_nucleus,
)
from wuyun.thalamus.thalamic_router import ThalamicRouter

__all__ = [
    "ThalamicNucleus",
    "create_thalamic_nucleus",
    "ThalamicRouter",
]