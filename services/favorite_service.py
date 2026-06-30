"""
============================================================
收藏业务服务(FavoriteService)
============================================================
【业务定位】用户对"穿搭方案 / 单品"的收藏
  - 支持"按签名去重":同一 source+key 的收藏只保留一份
  - 支持"按签名删除":前端点 ❤️ / 💔 切换
【数据表】user_favorites
【调用方】routes/favorite_routes.py
============================================================
"""

# ── 类型注解 ──
from typing import List, Dict, Any

# ── 数据访问层 ──
from database.models import DatabaseManager


class FavoriteService:
    """收藏业务服务"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()

    # ============================================================
    # 工具:把"穿搭组合 + 天气上下文"编码成一个稳定签名
    # ============================================================
    @staticmethod
    def _combo_signature(outfit_combo: List, weather_context: Dict[str, Any] = None) -> str:
        """
        签名算法:由 DB 层的 make_signature 统一实现
        同一套衣服 + 同一份天气 → 同一个签名 → 收藏时去重
        """
        return DatabaseManager.make_signature(outfit_combo, weather_context)

    # ============================================================
    # 收藏(带去重)
    # ============================================================
    def create_favorite(self, user_id: int, outfit_combo: List[int],
                       weather_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        收藏一个穿搭方案
        1) 先查已有收藏,计算新签名
        2) 命中已存在 → 返回 duplicate=True(不重复入库)
        3) 未命中 → 写入 + 返回新 favorite
        """
        try:
            # 1) 查已有
            existing     = self.db.get_user_favorites(user_id)
            new_sig      = self._combo_signature(outfit_combo, weather_context)

            # 2) 命中已有 → 直接返回,不重复入库
            for fav in existing:
                if self._combo_signature(fav.get('outfit_combo', []), fav.get('weather_context')) == new_sig:
                    return {'success': True, 'favorite': fav, 'duplicate': True}

            # 3) 写入
            favorite_id = self.db.create_favorite(
                user_id=user_id,
                outfit_combo=outfit_combo,
                weather_context=weather_context
            )
            favorites = self.db.get_user_favorites(user_id)
            favorite  = next((f for f in favorites if f['id'] == favorite_id), None)

            return {'success': True, 'favorite': favorite, 'duplicate': False}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 按签名删除(取消收藏)
    # ============================================================
    def delete_favorite_by_signature(self, user_id: int, signature: str) -> Dict[str, Any]:
        """前端点 💔 时按签名精准删除"""
        try:
            ok = self.db.delete_favorite_by_signature(user_id, signature)
            if ok:
                return {'success': True, 'message': '已取消收藏'}
            return {'success': False, 'error': '未找到该收藏'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 列出某用户所有收藏
    # ============================================================
    def get_user_favorites(self, user_id: int) -> Dict[str, Any]:
        """收藏页"我的收藏"模块用"""
        try:
            favorites = self.db.get_user_favorites(user_id)
            return {'success': True, 'favorites': favorites}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 按收藏 ID 删除(管理/单条删除)
    # ============================================================
    def delete_favorite(self, favorite_id: int) -> Dict[str, Any]:
        """按主键 ID 删除"""
        try:
            success = self.db.delete_favorite(favorite_id)
            if success:
                return {'success': True, 'message': 'Favorite deleted successfully'}
            return {'success': False, 'error': 'Favorite not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
