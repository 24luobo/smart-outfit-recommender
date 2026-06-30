"""
============================================================
项目配置(config.py)
============================================================
【作用】项目唯一的"配置中心",所有路径/参数都从这读
  - 数据库路径
  - 数据/模型/上传目录
  - 机器学习参数(K-Means 簇数、随机种子)

【使用方式】
  from config import Config
  db_path = Config.DATABASE_PATH

【优势】改一个地方,全应用生效
============================================================
"""

# ── 路径处理 ──
import os


class Config:
    """
    项目配置类(类属性即配置项)
    注意:类定义体里直接 makedirs,导入本模块时即创建好目录
    """

    # ── Flask 安全密钥(生产环境必须改) ──
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # ── 数据库路径(SQLite 单文件) ──
    # 取本文件所在目录的 database/outfit_recommender.db
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'database', 'outfit_recommender.db')

    # ── 数据目录:训练用 CSV 放这里 ──
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

    # ── 模型目录:训练产物 .pkl 放这里 ──
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

    # ── 上传目录:用户上传图片的保存位置 ──
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'images', 'uploads')

    # ── 启动时一次性创建所有必要目录(避免后续 makedirs 散落各处) ──
    for directory in [DATA_DIR, MODELS_DIR, UPLOAD_FOLDER]:
        os.makedirs(directory, exist_ok=True)

    # ── 机器学习参数 ──
    KMEANS_CLUSTERS = 8          # K-Means 簇数 = 8 大风格
    RANDOM_STATE    = 42         # 固定随机种子,保证结果可复现
