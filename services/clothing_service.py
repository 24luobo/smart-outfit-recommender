"""
============================================================
服饰业务服务(ClothingService)
============================================================
【业务定位】服饰(单品)的 CRUD 业务层
  - 在路由和 DB 之间做一道"业务校验 + 异常包装"
  - 每个方法返回 {'success': bool, '...': ..., 'error': str} 的统一格式
【数据表】clothing_items
【调用方】routes/clothing_routes.py
============================================================
"""

# ── 类型注解 ──
from typing import List, Dict, Any, Optional

# ── 数据访问层 ──
from database.models import DatabaseManager


class ClothingService:
    """服饰业务服务"""

    def __init__(self, db_manager: DatabaseManager = None):
        """构造方法,允许外部注入 db_manager 便于单元测试"""
        self.db = db_manager or DatabaseManager()

    # ============================================================
    # 创建服饰
    # ============================================================
    def create_clothing_item(self, name: str, category: str, color: str = None,
                            style: str = None, season: str = None,
                            temp_min: float = None, temp_max: float = None,
                            image_path: str = None, feature_vector: List[float] = None,
                            cluster_id: int = None) -> Dict[str, Any]:
        """
        新增一件服饰
        :return: {'success': True/False, 'item': {...}, 'error': str}
        """
        try:
            item_id = self.db.create_clothing_item(
                name=name, category=category, color=color, style=style,
                season=season, temp_min=temp_min, temp_max=temp_max,
                image_path=image_path, feature_vector=feature_vector,
                cluster_id=cluster_id
            )
            # 写完再读出来返回(带回自增 id、时间戳等)
            item = self.db.get_clothing_item(item_id)
            return {'success': True, 'item': item}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 单件服饰详情
    # ============================================================
    def get_clothing_item(self, item_id: int) -> Dict[str, Any]:
        """按 ID 取一件服饰的完整信息"""
        try:
            item = self.db.get_clothing_item(item_id)
            if item:
                return {'success': True, 'item': item}
            return {'success': False, 'error': 'Clothing item not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 全部服饰列表(服饰库页面用)
    # ============================================================
    def get_all_clothing_items(self) -> Dict[str, Any]:
        """服饰库总览(可能有几百件,前端要分页)"""
        try:
            items = self.db.get_all_clothing_items()
            return {'success': True, 'items': items}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 按类目筛选
    # ============================================================
    def get_clothing_by_category(self, category: str) -> Dict[str, Any]:
        """category:上衣 / 裤子 / 外套 / 连衣裙 / 裙子 / 鞋子 / 配饰"""
        try:
            items = self.db.get_clothing_by_category(category)
            return {'success': True, 'items': items}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 更新聚类 ID(KMeans 训练后调用)
    # ============================================================
    def update_clothing_cluster(self, item_id: int, cluster_id: int) -> Dict[str, Any]:
        """把 K-Means 算出的聚类 ID 写回 DB"""
        try:
            success = self.db.update_clothing_cluster(item_id, cluster_id)
            if success:
                item = self.db.get_clothing_item(item_id)
                return {'success': True, 'item': item}
            return {'success': False, 'error': 'Clothing item not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 更新特征向量(K-Means 训练后调用)
    # ============================================================
    def update_clothing_features(self, item_id: int, feature_vector: List[float]) -> Dict[str, Any]:
        """把 128 维图像特征写回 DB(冗余存储,避免每次实时重算)"""
        try:
            success = self.db.update_clothing_features(item_id, feature_vector)
            if success:
                item = self.db.get_clothing_item(item_id)
                return {'success': True, 'item': item}
            return {'success': False, 'error': 'Clothing item not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
