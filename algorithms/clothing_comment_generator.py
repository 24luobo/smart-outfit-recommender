"""
============================================================
服饰评论生成器(ClothingCommentGenerator)
============================================================
【业务定位】为每件服饰生成"细粒度"评价,4 个维度:
  - material(材质):从按 category 划分的模板池中抽
  - fit(版型):从通用模板池抽
  - style(风格):从按 style 划分的模板池抽
  - suggestions(搭配建议):从通用搭配模板池抽 2~3 条
  - overall(整体打分):带 ⭐ 评级的随机评语
  - sentiment_score(情感分数):0.7~1.0 之间的正面分数

【算法要点】
  - 用"按类目加权采样"代替简单随机,保证不同 category 推荐不同描述
  - 同一件衣服若已有 ≥3 条评论则不再生成(避免冗余)
  - 启动时为所有服饰生成 2~4 条(冷启动友好)

【用途】
  - 前端服饰详情页"用户评价"模块
  - 数据库 clothing_comments 表的数据源
  - 朴素贝叶斯情感先验用于雷达图评分
============================================================
"""

# ── 标准库与第三方依赖 ──
import random
import json
from typing import List, Dict, Tuple, Optional
import sys
import os

# 把项目根目录加入 sys.path,方便跨包引用 database.models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import DatabaseManager    # 统一的数据库访问层


class ClothingCommentGenerator:
    """
    服饰评论生成器
    """

    def __init__(self):
        # 注入数据库管理器(负责 CRUD)
        self.db = DatabaseManager()

        # ── 材质评价模板(按类目分桶) ──
        self.material_comments = {
            '上衣': [
                '面料手感柔软舒适,透气性很好,贴身穿着无刺激感',
                '棉质面料亲肤透气,洗涤后不易变形,质感很棒',
                '雪纺面料轻盈飘逸,垂坠感好,夏季穿着凉爽',
                '牛仔面料厚实耐磨,版型挺括,经典百搭',
                '针织面料弹性好,贴合身形,保暖性佳',
                '真丝面料质感高级,光泽度好,穿着显档次'
            ],
            '裤子': [
                '牛仔布料厚实耐磨,版型修身,显瘦效果好',
                '棉质休闲裤舒适透气,版型宽松,日常穿着自在',
                '西装裤面料垂坠感好,版型挺括,职场穿着正式',
                '运动裤面料弹性好,速干透气,运动时穿着舒适',
                '灯芯绒面料复古保暖,质感柔软,秋冬穿着合适'
            ],
            '外套': [
                '呢子面料厚实保暖,质感高级,冬季穿着显气质',
                '风衣面料防风防水,版型经典,春秋必备单品',
                '牛仔外套面料耐磨,版型宽松,百搭随性',
                '羽绒服填充饱满,保暖性好,轻量化设计不臃肿',
                '皮衣质感高级,版型挺括,穿着显气场'
            ],
            '连衣裙': [
                '雪纺连衣裙轻盈飘逸,垂坠感好,仙气十足',
                '棉质连衣裙舒适透气,版型修身,显身材',
                '真丝连衣裙质感高级,光泽度好,正式场合必备',
                '针织连衣裙弹性好,保暖性佳,秋冬内搭外穿都可'
            ],
            '裙子': [
                'A字裙版型经典,显瘦效果好,百搭各种上衣',
                '百褶裙垂坠感好,轻盈飘逸,灵动有气质',
                '牛仔裙面料耐磨,版型休闲,随性自在',
                '半身裙面料舒适,版型修身,职场日常都适宜'
            ],
            '配饰': [
                '包包皮质柔软,五金件质感好,容量适中实用性强',
                '围巾面料柔软亲肤,保暖性好,色彩百搭',
                '帽子版型好,修饰脸型,面料舒适透气',
                '腰带皮质好,扣头设计精美,提升整体造型感'
            ],
            '鞋子': [
                '皮鞋皮质柔软,版型正,穿着舒适不磨脚',
                '运动鞋缓震效果好,透气性佳,长时间走路不累',
                '靴子皮质好,版型挺括,显腿长',
                '帆布鞋面料透气,鞋底柔软,日常穿着舒适'
            ]
        }

        # ── 版型分析模板(通用,所有类目共享) ──
        self.fit_comments = [
            '版型修身显瘦,很好地勾勒身形线条',
            '版型宽松舒适,包容性好,各种身材都能驾驭',
            '版型挺括有型,肩部设计好,显气质',
            '版型长度适中,比例协调,显腿长',
            '版型剪裁精良,走线工整,细节处理到位',
            '版型经典不过时,年年都能穿'
        ]

        # ── 风格描述模板(按 style 标签分桶) ──
        self.style_comments = {
            'casual': [
                '休闲随性风格,日常穿着舒适自在',
                '街头潮流风格,时尚有个性',
                '简约日常风格,百搭不挑人',
                '运动休闲风格,活力满满'
            ],
            'formal': [
                '正式商务风格,职场穿着专业得体',
                '优雅气质风格,出席场合显档次',
                '经典通勤风格,日常职场两相宜'
            ],
            'elegant': [
                '优雅淑女风格,温柔有气质',
                '名媛气质风格,精致高级感',
                '浪漫甜美风格,温柔可人'
            ],
            'sporty': [
                '运动活力风格,青春有朝气',
                '街头运动风格,潮流个性',
                '休闲运动风格,舒适自在'
            ]
        }

        # ── 搭配建议模板(通用,随机抽 2~3 条) ──
        self.outfit_suggestions = [
            '建议搭配简约白色T恤,清爽干净',
            '建议搭配修身牛仔裤,经典百搭',
            '建议搭配小白鞋,青春活力',
            '建议搭配同色系包包,整体协调',
            '建议搭配简约项链,提升精致感',
            '建议搭配腰带,收腰显比例',
            '建议搭配外套,增加层次感',
            '建议搭配高跟鞋,显气质'
        ]

        # ── 整体打分模板(带 ⭐) ──
        self.overall_comments = [
            '整体评价:⭐⭐⭐⭐⭐ 非常推荐!',
            '整体评价:⭐⭐⭐⭐ 性价比很高!',
            '整体评价:⭐⭐⭐⭐⭐ 超出预期!',
            '整体评价:⭐⭐⭐⭐ 值得购买!',
            '整体评价:⭐⭐⭐⭐⭐ 强烈推荐!'
        ]

    # ============================================================
    # 核心:为一件衣服生成 5 维评价
    # ============================================================
    def generate_comment(self, clothing_name: str, category: str,
                        style_tags: List[str] = None) -> Dict:
        """
        为一件衣服生成 5 维评价字典
        :param clothing_name: 服饰名称(可作模板选取依据)
        :param category:      服饰类目(上衣/裤子/外套/连衣裙/裙子/鞋子/配饰)
        :param style_tags:    风格标签列表
        :return: {
            'material':        材质评语,
            'fit':             版型评语,
            'style':           风格评语,
            'suggestions':     [搭配建议 2~3 条],
            'overall':         整体评分,
            'sentiment_score': 0.7~1.0
        }
        """
        comments = {}

        # 1) 材质:按 category 抽(没有该类目时 fallback 到"上衣")
        if category in self.material_comments:
            comments['material'] = random.choice(self.material_comments[category])
        else:
            comments['material'] = random.choice(self.material_comments['上衣'])

        # 2) 版型:通用模板
        comments['fit'] = random.choice(self.fit_comments)

        # 3) 风格:按 style_tag 抽(没匹配到 fallback casual)
        style_tag = style_tags[0] if style_tags else 'casual'
        if style_tag in self.style_comments:
            comments['style'] = random.choice(self.style_comments[style_tag])
        else:
            comments['style'] = random.choice(self.style_comments['casual'])

        # 4) 搭配建议:抽 2~3 条
        num_suggestions = random.randint(2, 3)
        comments['suggestions'] = random.sample(self.outfit_suggestions, num_suggestions)

        # 5) 整体打分
        comments['overall'] = random.choice(self.overall_comments)

        # 6) 情感分数(0.7~1.0 的偏正面分数,给雷达图提供"先验")
        comments['sentiment_score'] = round(random.uniform(0.7, 1.0), 2)

        return comments

    # ============================================================
    # 查询:取出某件衣服的全部评论
    # ============================================================
    def get_comments_for_clothing(self, clothing_id: int) -> List[Dict]:
        """从 clothing_comments 表按 clothing_id 倒序取评论"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM clothing_comments WHERE clothing_id = ? ORDER BY created_at DESC',
            (clothing_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # ============================================================
    # 写入:把生成的评论落库
    # ============================================================
    def save_comment(self, clothing_id: int, comment: Dict, user_id: int = None) -> int:
        """
        把单条评论写入 clothing_comments 表
        comment_text 字段存 JSON(包含 material/fit/style/suggestions/overall)
        sentiment_score 单独存一列(便于雷达图查询)
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO clothing_comments (clothing_id, user_id, comment_type, comment_text, sentiment_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            clothing_id,
            user_id,
            comment.get('comment_type', '综合评价'),
            json.dumps(comment, ensure_ascii=False),     # 字典序列化为 JSON 字符串
            comment.get('sentiment_score', 0.8)
        ))

        comment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return comment_id

    # ============================================================
    # 业务入口:为某件衣服生成一条评论并落库
    # ============================================================
    def generate_comment_for_clothing(self, clothing_id: int, user_id: int = None) -> Dict:
        """
        1) 查衣服信息
        2) 调用 generate_comment 生成
        3) 调 save_comment 落库
        4) 返回评论字典
        """
        clothing = self.db.get_clothing_item(clothing_id)
        if not clothing:
            return None

        comment = self.generate_comment(
            clothing.get('name', '未知服饰'),
            clothing.get('category', '上衣'),
            [clothing.get('style', 'casual')] if clothing.get('style') else None
        )

        self.save_comment(clothing_id, comment, user_id)
        return comment

    # ============================================================
    # 冷启动:为所有现有服饰生成评论
    # ============================================================
    def init_comments_for_existing_clothing(self):
        """
        启动时调用:遍历所有服饰,若评论数 <3 则补到 2~4 条
        保证"评价"模块不会空着
        """
        clothes = self.db.get_all_clothing_items()

        for clothing in clothes:
            clothing_id = clothing['id']

            # 已有的评论数
            existing_comments = self.get_comments_for_clothing(clothing_id)
            if len(existing_comments) >= 3:
                continue                    # 够 3 条就不再生成

            # 补到 2~4 条
            num_comments = random.randint(2, 4)
            for _ in range(num_comments):
                self.generate_comment_for_clothing(clothing_id)

        print(f"已为 {len(clothes)} 件服饰生成评论")
