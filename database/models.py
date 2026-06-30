"""
============================================================
数据库访问层(DatabaseManager) - 全量 CRUD
============================================================
【作用】项目唯一的数据库入口,封装 7 张表的所有 CRUD
  - 每个方法 = 一次独立的 sqlite3 连接(短连接,够用)
  - 返回 dict(不是 Row),方便 JSON 序列化
  - JSON 字段(usual_scenes/feature_vector/outfit_combo/weather_context)自动反序列化

【设计原则】
  - 上层(services)不直接写 SQL,统一通过本类
  - 单元测试时可注入 db_path
  - 收藏去重的"签名算法"也放在这里,作为单一来源
============================================================
"""

# ── 标准库 ──
import sqlite3
import json
import os
# ── 类型注解 ──
from typing import List, Dict, Any, Optional
# ── 配置 ──
from config import Config


class DatabaseManager:
    """
    数据库访问层 - 7 张表的全部 CRUD
    """

    def __init__(self, db_path: str = None):
        """
        :param db_path: 数据库文件路径,默认从 Config 读
        """
        self.db_path = db_path or Config.DATABASE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    # ============================================================
    # 工具:获取短连接(每次都 new,用完 close)
    # ============================================================
    def get_connection(self):
        """
        新建一个 sqlite3 连接
        row_factory = Row → 可以用列名访问(也可以转 dict)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ============================================================
    # 用户相关 CRUD
    # ============================================================
    def create_user(self, height: float, skin_tone: str,
                   style_preference: str = None, usual_scenes: List[str] = None,
                   weight: float = None, username: str = None,
                   password_hash: str = None, is_guest: bool = False) -> int:
        """注册新用户/创建游客,返回 user_id"""
        conn = self.get_connection()
        cursor = conn.cursor()
        # usual_scenes 是 list → 存为 JSON
        scenes_json = json.dumps(usual_scenes) if usual_scenes else None

        cursor.execute('''
            INSERT INTO users (username, password_hash, height, weight, skin_tone,
                              style_preference, usual_scenes, is_guest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, password_hash, height, weight, skin_tone,
              style_preference, scenes_json, 1 if is_guest else 0))

        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """按用户名查(注册时查重)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()

        if row:
            user = dict(row)
            if user.get('usual_scenes'):
                user['usual_scenes'] = json.loads(user['usual_scenes'])
            return user
        return None

    def verify_user(self, username: str, password_hash: str) -> Optional[Dict[str, Any]]:
        """登录验证:用户名 + 哈希匹配 + 非游客"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM users
            WHERE username = ? AND password_hash = ? AND is_guest = 0
        ''', (username, password_hash))
        row = cursor.fetchone()
        conn.close()

        if row:
            user = dict(row)
            if user.get('usual_scenes'):
                user['usual_scenes'] = json.loads(user['usual_scenes'])
            return user
        return None

    def create_guest_user(self) -> int:
        """
        创建游客用户
        兼容老 schema(显式写入默认值,避免 NOT NULL 约束报错)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (is_guest, height, weight, skin_tone,
                               style_preference, usual_scenes)
            VALUES (1, 0, 0, '', '', '[]')
        ''')
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """按 ID 查用户"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            user = dict(row)
            if user.get('usual_scenes'):
                user['usual_scenes'] = json.loads(user['usual_scenes'])
            return user
        return None

    def update_user(self, user_id: int, **kwargs) -> bool:
        """
        局部更新用户档案(白名单字段防 SQL 注入)
        包含 usual_scenes 自动 JSON 序列化
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # 白名单字段(防止传入任意字段更新数据库)
        allowed_fields = ['height', 'weight', 'skin_tone', 'style_preference', 'usual_scenes']
        updates, values = [], []

        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == 'usual_scenes' and value:
                    value = json.dumps(value)        # list → JSON
                updates.append(f"{key} = ?")
                values.append(value)

        if updates:
            values.append(user_id)
            cursor.execute(f'UPDATE users SET {", ".join(updates)} WHERE id = ?', values)
            conn.commit()
            success = cursor.rowcount > 0
        else:
            success = False

        conn.close()
        return success

    def get_all_users(self) -> List[Dict[str, Any]]:
        """全量用户列表(管理后台/调试用)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        rows = cursor.fetchall()
        conn.close()

        users = []
        for row in rows:
            user = dict(row)
            if user.get('usual_scenes'):
                user['usual_scenes'] = json.loads(user['usual_scenes'])
            users.append(user)
        return users

    # ============================================================
    # 天气相关 CRUD
    # ============================================================
    def create_weather(self, temperature: float, weather_condition: str, season: str) -> int:
        """录入一条天气,返回 weather_id"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO weather (temperature, weather_condition, season)
            VALUES (?, ?, ?)
        ''', (temperature, weather_condition, season))
        weather_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return weather_id

    def get_weather(self, weather_id: int) -> Optional[Dict[str, Any]]:
        """按 ID 查天气"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM weather WHERE id = ?', (weather_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_latest_weather(self) -> Optional[Dict[str, Any]]:
        """取最近一条天气(首页默认显示)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM weather ORDER BY recorded_at DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # ============================================================
    # 服饰相关 CRUD
    # ============================================================
    def create_clothing_item(self, name: str, category: str, color: str = None,
                            style: str = None, season: str = None,
                            temp_min: float = None, temp_max: float = None,
                            image_path: str = None, feature_vector: List[float] = None,
                            cluster_id: int = None) -> int:
        """新增一件服饰,返回 item_id"""
        conn = self.get_connection()
        cursor = conn.cursor()
        feature_json = json.dumps(feature_vector) if feature_vector else None

        cursor.execute('''
            INSERT INTO clothing_items (name, category, color, style, season,
                                       suitable_temperature_min, suitable_temperature_max,
                                       image_path, feature_vector, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, category, color, style, season, temp_min, temp_max,
              image_path, feature_json, cluster_id))
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return item_id

    def get_clothing_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """按 ID 查服饰(feature_vector 自动反序列化)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM clothing_items WHERE id = ?', (item_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            item = dict(row)
            if item.get('feature_vector'):
                item['feature_vector'] = json.loads(item['feature_vector'])
            return item
        return None

    def get_all_clothing_items(self) -> List[Dict[str, Any]]:
        """全量服饰列表(服饰库)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM clothing_items')
        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            item = dict(row)
            if item.get('feature_vector'):
                item['feature_vector'] = json.loads(item['feature_vector'])
            items.append(item)
        return items

    def get_clothing_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类目筛选服饰"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM clothing_items WHERE category = ?', (category,))
        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            item = dict(row)
            if item.get('feature_vector'):
                item['feature_vector'] = json.loads(item['feature_vector'])
            items.append(item)
        return items

    def update_clothing_cluster(self, item_id: int, cluster_id: int) -> bool:
        """KMeans 训练后,把聚类 ID 写回"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE clothing_items SET cluster_id = ? WHERE id = ?',
                      (cluster_id, item_id))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def update_clothing_features(self, item_id: int, feature_vector: List[float]) -> bool:
        """把 128 维图像特征写回(冗余存储)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        feature_json = json.dumps(feature_vector)
        cursor.execute('UPDATE clothing_items SET feature_vector = ? WHERE id = ?',
                      (feature_json, item_id))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    # ============================================================
    # 收藏相关 CRUD
    # ============================================================
    def create_favorite(self, user_id: int, outfit_combo: List[int],
                       weather_context: Dict[str, Any] = None) -> int:
        """收藏一个穿搭方案"""
        conn = self.get_connection()
        cursor = conn.cursor()
        outfit_json  = json.dumps(outfit_combo)
        weather_json = json.dumps(weather_context) if weather_context else None

        cursor.execute('''
            INSERT INTO user_favorites (user_id, outfit_combo, weather_context)
            VALUES (?, ?, ?)
        ''', (user_id, outfit_json, weather_json))
        favorite_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return favorite_id

    def get_user_favorites(self, user_id: int) -> List[Dict[str, Any]]:
        """查某用户所有收藏(按时间倒序),JSON 自动反序列化"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_favorites WHERE user_id = ? ORDER BY created_at DESC',
                      (user_id,))
        rows = cursor.fetchall()
        conn.close()

        favorites = []
        for row in rows:
            fav = dict(row)
            fav['outfit_combo'] = json.loads(fav['outfit_combo'])
            if fav.get('weather_context'):
                fav['weather_context'] = json.loads(fav['weather_context'])
            favorites.append(fav)
        return favorites

    def delete_favorite(self, favorite_id: int) -> bool:
        """按主键 ID 删除"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_favorites WHERE id = ?', (favorite_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def find_favorite_by_signature(self, user_id: int, signature: str):
        """
        按签名找收藏的 ID(用于"按签名取消收藏")
        签名 = f"{source}::{key}"(由 make_signature 生成)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, outfit_combo, weather_context FROM user_favorites
            WHERE user_id = ? ORDER BY created_at DESC
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()
        # 在 Python 端算签名对比(SQLite 不方便做 JSON 字段对比)
        for row in rows:
            fav = dict(row)
            try:
                combo = json.loads(fav['outfit_combo']) if fav['outfit_combo'] else []
            except Exception:
                combo = []
            try:
                ctx = json.loads(fav['weather_context']) if fav['weather_context'] else {}
            except Exception:
                ctx = {}
            if DatabaseManager.make_signature(combo, ctx) == signature:
                return fav['id']
        return None

    def delete_favorite_by_signature(self, user_id: int, signature: str) -> bool:
        """先找 ID,再删;找不到返回 False"""
        fid = self.find_favorite_by_signature(user_id, signature)
        if fid is None:
            return False
        return self.delete_favorite(fid)

    @staticmethod
    def make_signature(outfit_combo, weather_context):
        """
        生成去重签名(收藏 / 取消收藏时用)
        - source='outfit_card' 时 → 用 image URL(或第一件 ID)
        - 其他 → 用排序后的 ID 列表拼接
        """
        ctx = weather_context or {}
        source = ctx.get('source', '')
        if source == 'outfit_card':
            key = ctx.get('image', '') or (outfit_combo[0] if outfit_combo else '')
        else:
            key = '|'.join(str(x) for x in sorted(outfit_combo or []))
        return f'{source}::{key}'

    # ============================================================
    # 推荐历史 CRUD
    # ============================================================
    def create_recommendation_history(self, user_id: int, weather_id: int,
                                     recommended_outfits: List[Dict[str, Any]],
                                     feedback: int = None) -> int:
        """把一次推荐结果入库(供"历史"模块查询)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        outfits_json = json.dumps(recommended_outfits)
        cursor.execute('''
            INSERT INTO recommendation_history (user_id, weather_id, recommended_outfits, feedback)
            VALUES (?, ?, ?, ?)
        ''', (user_id, weather_id, outfits_json, feedback))
        history_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return history_id

    def get_recommendation_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """查某用户最近 N 条推荐历史"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM recommendation_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            h = dict(row)
            h['recommended_outfits'] = json.loads(h['recommended_outfits'])
            history.append(h)
        return history

    def update_recommendation_feedback(self, history_id: int, feedback: int) -> bool:
        """用户对一次推荐点赞/踩:feedback ∈ {0, 1}"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE recommendation_history SET feedback = ? WHERE id = ?',
                      (feedback, history_id))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    # ============================================================
    # 身材信息 CRUD
    # ============================================================
    def create_body_shape(self, user_id: int, **kwargs) -> int:
        """
        一次性插入全套身材参数
        白名单 allowed_fields 防止非法字段写入
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        allowed_fields = [
            'height', 'head_circumference', 'shoulder_width',
            'chest_circumference', 'waist_circumference', 'abdomen_circumference',
            'forearm_length', 'arm_length', 'hip_circumference',
            'wrist_circumference', 'thigh_length', 'calf_length',
            'foot_length', 'weight', 'skin_tone', 'body_type', 'gender'
        ]
        fields, values = [], []
        for key, value in kwargs.items():
            if key in allowed_fields:
                fields.append(key)
                values.append(value)
        placeholders = ', '.join(['?' for _ in values])
        field_names  = ', '.join(fields)
        cursor.execute(f'''
            INSERT INTO body_shape (user_id, {field_names})
            VALUES (?, {placeholders})
        ''', [user_id] + values)
        body_shape_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return body_shape_id

    def get_body_shape(self, user_id: int) -> Optional[Dict[str, Any]]:
        """查某用户最新一条身材数据(按 updated_at 倒序)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM body_shape WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1',
                      (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_body_shape(self, user_id: int, **kwargs) -> bool:
        """
        upsert 身材数据:
          - 已有 → UPDATE
          - 没有 → INSERT
        白名单字段 + 自动更新 updated_at
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        # 先查是否已有
        cursor.execute('SELECT id FROM body_shape WHERE user_id = ?', (user_id,))
        existing = cursor.fetchone()

        allowed_fields = [
            'height', 'head_circumference', 'shoulder_width',
            'chest_circumference', 'waist_circumference', 'abdomen_circumference',
            'forearm_length', 'arm_length', 'hip_circumference',
            'wrist_circumference', 'thigh_length', 'calf_length',
            'foot_length', 'weight', 'skin_tone', 'body_type', 'gender'
        ]
        updates, values = [], []
        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                values.append(value)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            if existing:
                values.append(user_id)
                cursor.execute(f'UPDATE body_shape SET {", ".join(updates)} WHERE user_id = ?', values)
            else:
                # 没有就 INSERT
                field_names  = ', '.join([k for k in kwargs.keys() if k in allowed_fields])
                placeholders = ', '.join(['?' for _ in values])
                cursor.execute(f'''
                    INSERT INTO body_shape (user_id, {field_names}, updated_at)
                    VALUES (?, {placeholders}, CURRENT_TIMESTAMP)
                ''', [user_id] + values)
            conn.commit()
            success = True
        else:
            success = False
        conn.close()
        return success

    # ============================================================
    # 浏览历史 CRUD
    # ============================================================
    def create_browse_history(self, user_id: int, item_type: str, item_id: int = None,
                              title: str = None, image_url: str = None,
                              meta: Dict[str, Any] = None) -> int:
        """记录一次浏览事件"""
        conn = self.get_connection()
        cursor = conn.cursor()
        meta_json = json.dumps(meta) if meta else None
        cursor.execute('''
            INSERT INTO browse_history (user_id, item_type, item_id, title, image_url, meta)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, item_type, item_id, title, image_url, meta_json))
        history_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return history_id

    def get_browse_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """查某用户最近 N 条浏览记录(meta 字段 JSON 自动反序列化)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM browse_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        result = []
        for row in rows:
            h = dict(row)
            if h.get('meta'):
                try:
                    h['meta'] = json.loads(h['meta'])
                except Exception:
                    pass
            result.append(h)
        return result

    def delete_browse_history(self, history_id: int) -> bool:
        """单条删除"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM browse_history WHERE id = ?', (history_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def clear_browse_history(self, user_id: int) -> int:
        """清空某用户全部历史,返回删除条数"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM browse_history WHERE user_id = ?', (user_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted
