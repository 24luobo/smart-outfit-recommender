"""
============================================================
数据库初始化(database/init_db.py)
============================================================
【作用】项目启动时建库 + 建表 + 幂等迁移
  - 7 张表:users / body_shape / weather / clothing_items /
            user_favorites / recommendation_history / browse_history
  - 幂等迁移:对已存在的旧库,自动补齐缺失列(不删数据)

【调用时机】
  - app.py 启动时 init_database()
  - 手动:python -m database.init_db
============================================================
"""

# ── SQLite 驱动 ──
import sqlite3
# ── 路径处理 ──
import os
# ── 项目配置(数据库路径) ──
from config import Config


def init_database(db_path: str = None):
    """
    建库 + 建表 + 幂等迁移

    设计原则:
      - 所有表 IF NOT EXISTS,已存在则跳过
      - 字段命名清晰,符合业务语义
      - 外键约束保证数据完整性
    """
    db_path = db_path or Config.DATABASE_PATH
    # 确保 database 目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ────────────────────────────────────────────────
    # 1) 用户表 users
    #    字段:账号体系(username + password_hash) + 身材/偏好(height/weight/skin_tone/style_preference)
    #    + 常用场景(usual_scenes,JSON) + 是否游客(is_guest)
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE,            -- 用户名(注册用,游客为 NULL)
            password_hash VARCHAR(255),             -- SHA256 密码哈希
            height REAL,                            -- 身高 cm
            weight REAL,                            -- 体重 kg
            skin_tone VARCHAR(20),                  -- 肤色(fair/medium/dark/olive)
            style_preference VARCHAR(100),           -- 偏好风格
            usual_scenes TEXT,                      -- 常用场景(JSON 列表)
            is_guest BOOLEAN DEFAULT 0,             -- 是否游客(1=是,0=否)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ────────────────────────────────────────────────
    # 2) 身材信息表 body_shape
    #    字段:全套身材参数(头围/肩宽/胸围/腰围/腹围/前臂长/臂长/臀围/腕围/大腿长/小腿长/足长)
    #    + 体重/肤色/体型(gender/body_type)
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS body_shape (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            height REAL,                            -- 身高
            head_circumference REAL,                -- 头围
            shoulder_width REAL,                    -- 肩宽
            chest_circumference REAL,               -- 胸围
            waist_circumference REAL,               -- 腰围
            abdomen_circumference REAL,             -- 腹围
            forearm_length REAL,                    -- 前臂长
            arm_length REAL,                        -- 臂长
            hip_circumference REAL,                 -- 臀围
            wrist_circumference REAL,               -- 腕围
            thigh_length REAL,                      -- 大腿长
            calf_length REAL,                       -- 小腿长
            foot_length REAL,                       -- 足长
            weight REAL,                            -- 体重
            skin_tone VARCHAR(20),                  -- 肤色
            body_type VARCHAR(30),                  -- 体型(瘦/标准/微胖/...)
            gender VARCHAR(10) DEFAULT 'female',    -- 性别
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ────────────────────────────────────────────────
    # 3) 天气表 weather
    #    朴素贝叶斯的输入
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL NOT NULL,              -- 温度 ℃
            weather_condition VARCHAR(20) NOT NULL, -- 天气(sunny/rainy/snowy/cloudy)
            season VARCHAR(20) NOT NULL,            -- 季节(spring/summer/autumn/winter)
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ────────────────────────────────────────────────
    # 4) 服饰表 clothing_items
    #    K-Means 聚类对象;feature_vector 存 128 维图像特征(JSON 字符串)
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothing_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,             -- 服饰名称
            category VARCHAR(50) NOT NULL,          -- 类目(上衣/裤子/外套/连衣裙/裙子/鞋子/配饰)
            color VARCHAR(50),                      -- 颜色
            style VARCHAR(50),                      -- 风格
            season VARCHAR(20),                     -- 季节(all 表示全年)
            suitable_temperature_min REAL,          -- 适用最低温
            suitable_temperature_max REAL,          -- 适用最高温
            image_path VARCHAR(255),                -- 图片路径
            feature_vector TEXT,                    -- 128 维特征向量(JSON)
            cluster_id INTEGER,                     -- K-Means 聚类 ID(0~7)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ────────────────────────────────────────────────
    # 5) 用户收藏表 user_favorites
    #    outfit_combo 存穿搭组合(JSON 列表)
    #    weather_context 存当时的天气(JSON,用于去重)
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            outfit_combo TEXT NOT NULL,             -- 穿搭组合(JSON)
            weather_context TEXT,                   -- 当时天气(JSON)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ────────────────────────────────────────────────
    # 6) 推荐历史表 recommendation_history
    #    recommended_outfits 存 5 套穿搭方案(JSON)
    #    feedback 存用户反馈(1=赞 / 0=踩)
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            weather_id INTEGER NOT NULL,
            recommended_outfits TEXT NOT NULL,
            feedback INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (weather_id) REFERENCES weather(id)
        )
    ''')

    # ────────────────────────────────────────────────
    # 7) 浏览历史表 browse_history
    #    用户点开服饰/穿搭方案时记录
    #    item_type 区分 outfit / clothing
    #    meta 存任意 JSON 元数据
    # ────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS browse_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            item_type VARCHAR(20) NOT NULL,         -- 'outfit' | 'clothing'
            item_id INTEGER,                        -- 对应 ID(穿搭方案可空)
            title VARCHAR(200),                     -- 展示标题
            image_url VARCHAR(500),                 -- 缩略图 URL
            meta TEXT,                              -- JSON 元数据
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    # 加索引,加快按用户+时间查询
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_browse_user_time ON browse_history(user_id)')

    conn.commit()

    # ============================================================
    # 幂等迁移:为已存在的旧库补齐缺失列(不删数据)
    # ============================================================
    def _ensure_column(table, col, decl):
        """
        检查表里是否已有某列,没有就 ALTER TABLE 加上
        用于项目升级时新增字段,保证老库不报错
        """
        cursor.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cursor.fetchall()}
        if col not in cols:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                print(f"  - Migrated: added {table}.{col}")
            except Exception as e:
                print(f"  ! Migration failed for {table}.{col}: {e}")

    # 老 users 表可能没有这些列,统统补上
    _ensure_column('users', 'is_guest',          'BOOLEAN DEFAULT 0')
    _ensure_column('users', 'weight',            'REAL')
    _ensure_column('users', 'style_preference',  'VARCHAR(100)')
    _ensure_column('users', 'usual_scenes',      'TEXT')

    conn.commit()
    conn.close()

    print(f"Database initialized at {db_path}")


# 允许直接 python -m database.init_db 运行
if __name__ == '__main__':
    init_database()
