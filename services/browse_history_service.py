"""
============================================================
浏览历史业务服务(BrowseHistoryService)
============================================================
【业务定位】记录用户浏览过的"穿搭方案 / 单品"
  - 用于"个人中心 → 历史浏览"模块
  - 用于推荐算法做"基于历史的个性化"
【数据表】browse_history
【调用方】routes/browse_history_routes.py
============================================================
"""

# ── 类型注解 ──
from typing import Dict, Any, List

# ── 数据访问层 ──
from database.models import DatabaseManager


class BrowseHistoryService:
    """浏览历史业务服务"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()

    # ============================================================
    # 记录一次浏览
    # ============================================================
    def record(self, user_id: int, item_type: str, item_id: int = None,
               title: str = None, image_url: str = None,
               meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        记录一次浏览行为
        :param user_id:   用户 ID
        :param item_type: 'outfit'(穿搭方案) | 'clothing'(单品)
        :param item_id:   对应 ID(可空,穿搭方案可能没有 ID)
        :param title:     展示标题
        :param image_url: 缩略图
        :param meta:      任意 JSON 元数据
        """
        try:
            if not user_id or not item_type:
                return {'success': False, 'error': 'user_id 和 item_type 必填'}
            hid = self.db.create_browse_history(
                user_id=user_id, item_type=item_type, item_id=item_id,
                title=title, image_url=image_url, meta=meta
            )
            return {'success': True, 'history_id': hid}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 查询某用户的浏览历史
    # ============================================================
    def list_history(self, user_id: int, limit: int = 50) -> Dict[str, Any]:
        """按时间倒序返回最近 N 条"""
        try:
            items = self.db.get_browse_history(user_id, limit)
            return {'success': True, 'history': items, 'count': len(items)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 单条删除
    # ============================================================
    def delete(self, history_id: int) -> Dict[str, Any]:
        """按主键 ID 删除一条历史"""
        try:
            ok = self.db.delete_browse_history(history_id)
            return {'success': ok, 'message': '已删除' if ok else '记录不存在'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 清空
    # ============================================================
    def clear(self, user_id: int) -> Dict[str, Any]:
        """一键清空某用户所有历史"""
        try:
            n = self.db.clear_browse_history(user_id)
            return {'success': True, 'deleted': n}
        except Exception as e:
            return {'success': False, 'error': str(e)}
