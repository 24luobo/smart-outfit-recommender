"""
============================================================
用户业务服务(UserService)
============================================================
【业务定位】用户档案的 CRUD
  - 真实登录由 auth_routes 处理(账号密码/JWT)
  - 本服务负责"用户档案"维度的身高/体重/肤色/风格偏好/常用场景
【数据表】users
【调用方】routes/user_routes.py、auth_routes.py
============================================================
"""

# ── 类型注解 ──
from typing import List, Dict, Any, Optional

# ── 数据访问层 ──
from database.models import DatabaseManager


class UserService:
    """用户业务服务"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()

    # ============================================================
    # 创建用户档案
    # ============================================================
    def create_user(self, height: float, skin_tone: str,
                   style_preference: str = None, usual_scenes: List[str] = None,
                   weight: float = None) -> Dict[str, Any]:
        """
        新建用户档案
        :return: {'success': True, 'user': {...}}
        """
        try:
            user_id = self.db.create_user(
                height=height,
                skin_tone=skin_tone,
                style_preference=style_preference,
                usual_scenes=usual_scenes,
                weight=weight
            )
            user = self.db.get_user(user_id)
            return {'success': True, 'user': user}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 按 ID 查用户
    # ============================================================
    def get_user(self, user_id: int) -> Dict[str, Any]:
        """按 ID 取一份用户档案"""
        try:
            user = self.db.get_user(user_id)
            if user:
                return {'success': True, 'user': user}
            return {'success': False, 'error': 'User not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 更新用户档案(部分字段)
    # ============================================================
    def update_user(self, user_id: int, **kwargs) -> Dict[str, Any]:
        """
        局部更新:只传需要改的字段
        例:update_user(1, height=172, style_preference='sporty')
        """
        try:
            success = self.db.update_user(user_id, **kwargs)
            if success:
                user = self.db.get_user(user_id)
                return {'success': True, 'user': user}
            return {'success': False, 'error': 'User not found or no changes made'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 全量用户列表(管理后台用)
    # ============================================================
    def get_all_users(self) -> Dict[str, Any]:
        """管理后台 / 调试时调用"""
        try:
            users = self.db.get_all_users()
            return {'success': True, 'users': users}
        except Exception as e:
            return {'success': False, 'error': str(e)}
