"""
============================================================
示例数据初始化脚本(scripts/init_sample_data.py)
============================================================
【作用】首次部署时,向数据库写入 20 件示例服饰 + 3 个示例用户
  - 用于"冷启动"演示,启动后即可看到服饰库/用户数据
  - 不会重复写(可多次执行,但数据集会越来越大)

【使用方式】
  python scripts/init_sample_data.py

【注意】此脚本只插入最基础的演示数据,真实服饰库由
  data/sample_clothing_items.csv 批量导入(初始化时执行)
============================================================
"""

# ── 把项目根目录加入 sys.path,这样可以直接 import database.models ──
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 数据库访问 ──
from database.models import DatabaseManager
from database.init_db import init_database


def init_sample_data():
    """
    主流程:
      1) 建库建表
      2) 写 20 件示例服饰
      3) 写 3 个示例用户
    """
    # 1) 确保表结构存在
    init_database()
    db = DatabaseManager()

    # ────────────────────────────────────────────────
    # 2) 20 件示例服饰(覆盖 7 大类目 + 5 种风格)
    #    temp_min/temp_max:适用温度区间
    # ────────────────────────────────────────────────
    sample_clothing = [
        {'name': '白色T恤',     'category': 'top',        'color': 'white', 'style': 'casual',  'season': 'summer', 'temp_min': 20, 'temp_max': 35},
        {'name': '蓝色牛仔裤',   'category': 'bottom',     'color': 'blue',  'style': 'casual',  'season': 'all',    'temp_min': 10, 'temp_max': 30},
        {'name': '黑色西装外套', 'category': 'outerwear',  'color': 'black', 'style': 'formal',  'season': 'autumn', 'temp_min': 5,  'temp_max': 20},
        {'name': '红色连衣裙',   'category': 'dress',      'color': 'red',   'style': 'elegant', 'season': 'spring', 'temp_min': 15, 'temp_max': 28},
        {'name': '运动鞋',       'category': 'shoes',      'color': 'white', 'style': 'sporty',  'season': 'all',    'temp_min': -10,'temp_max': 35},
        {'name': '灰色卫衣',     'category': 'top',        'color': 'gray',  'style': 'casual',  'season': 'autumn', 'temp_min': 10, 'temp_max': 20},
        {'name': '黑色休闲裤',   'category': 'bottom',     'color': 'black', 'style': 'casual',  'season': 'all',    'temp_min': 10, 'temp_max': 30},
        {'name': '羽绒服',       'category': 'outerwear',  'color': 'black', 'style': 'casual',  'season': 'winter', 'temp_min': -15,'temp_max': 10},
        {'name': '粉色连衣裙',   'category': 'dress',      'color': 'pink',  'style': 'elegant', 'season': 'summer', 'temp_min': 20, 'temp_max': 35},
        {'name': '皮鞋',         'category': 'shoes',      'color': 'brown', 'style': 'formal',  'season': 'all',    'temp_min': 5,  'temp_max': 30},
        {'name': '白色衬衫',     'category': 'top',        'color': 'white', 'style': 'formal',  'season': 'spring', 'temp_min': 15, 'temp_max': 28},
        {'name': '藏青色西裤',   'category': 'bottom',     'color': 'navy',  'style': 'formal',  'season': 'all',    'temp_min': 10, 'temp_max': 30},
        {'name': '运动外套',     'category': 'outerwear',  'color': 'gray',  'style': 'sporty',  'season': 'autumn', 'temp_min': 8,  'temp_max': 20},
        {'name': '运动T恤',     'category': 'top',        'color': 'red',   'style': 'sporty',  'season': 'summer', 'temp_min': 20, 'temp_max': 35},
        {'name': '瑜伽裤',       'category': 'bottom',     'color': 'black', 'style': 'sporty',  'season': 'all',    'temp_min': 15, 'temp_max': 35},
        {'name': '风衣',         'category': 'outerwear',  'color': 'beige', 'style': 'elegant', 'season': 'spring', 'temp_min': 10, 'temp_max': 20},
        {'name': '毛衣',         'category': 'top',        'color': 'gray',  'style': 'casual',  'season': 'winter', 'temp_min': 0,  'temp_max': 15},
        {'name': '围巾',         'category': 'accessory',  'color': 'red',   'style': 'elegant', 'season': 'winter', 'temp_min': -10,'temp_max': 10},
        {'name': '帽子',         'category': 'accessory',  'color': 'black', 'style': 'casual',  'season': 'all',    'temp_min': -10,'temp_max': 35},
        {'name': '手链',         'category': 'accessory',  'color': 'gold',  'style': 'elegant', 'season': 'all',    'temp_min': 10, 'temp_max': 35},
    ]

    # 逐条写入
    for item in sample_clothing:
        db.create_clothing_item(
            name=item['name'],
            category=item['category'],
            color=item['color'],
            style=item['style'],
            season=item['season'],
            temp_min=item['temp_min'],
            temp_max=item['temp_max']
        )
    print(f'已添加 {len(sample_clothing)} 件示例服饰到数据库')

    # ────────────────────────────────────────────────
    # 3) 3 个示例用户(高矮胖瘦各一,不同肤色/风格)
    # ────────────────────────────────────────────────
    sample_users = [
        {
            'height': 175, 'weight': 68, 'skin_tone': 'medium',
            'style_preference': 'casual', 'usual_scenes': ['daily', 'work']
        },
        {
            'height': 165, 'weight': 52, 'skin_tone': 'fair',
            'style_preference': 'elegant', 'usual_scenes': ['work', 'party', 'date']
        },
        {
            'height': 180, 'weight': 75, 'skin_tone': 'dark',
            'style_preference': 'sporty', 'usual_scenes': ['sports', 'daily']
        },
    ]

    for user in sample_users:
        db.create_user(
            height=user['height'],
            weight=user['weight'],
            skin_tone=user['skin_tone'],
            style_preference=user['style_preference'],
            usual_scenes=user['usual_scenes']
        )
    print(f'已添加 {len(sample_users)} 个示例用户到数据库')
    print('示例数据初始化完成！')


# 允许 python scripts/init_sample_data.py 直接运行
if __name__ == '__main__':
    init_sample_data()
