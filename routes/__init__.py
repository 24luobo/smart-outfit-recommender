"""
============================================================
routes 包初始化文件
============================================================
作用:统一对外暴露所有 Flask 蓝图(Blueprint),app.py 一次性注册
每个 _bp 文件对应一组相关接口(用户/服饰/推荐/收藏/身材/认证/历史)
============================================================
"""

# ── 5 大基础模块的蓝图 ──
from .user_routes          import user_bp            # /api/users/...
from .clothing_routes      import clothing_bp        # /api/clothing/...
from .recommendation_routes import recommendation_bp  # /api/recommendations/...
from .favorite_routes      import favorite_bp        # /api/favorites/...
from .body_shape_routes    import body_shape_bp      # /api/body-shape/...

# ── `__all__` 控制 `from routes import *` 的导出范围 ──
__all__ = [
    'user_bp',
    'clothing_bp',
    'recommendation_bp',
    'favorite_bp',
    'body_shape_bp'
]
