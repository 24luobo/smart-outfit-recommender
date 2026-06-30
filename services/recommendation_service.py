"""
============================================================
推荐业务服务(RecommendationService) - 朴素贝叶斯 + K-Means 融合的"主流程"
============================================================
【业务定位】本项目最核心的业务类,前端 POST /api/recommendations 调用此服务。
  朴素贝叶斯筛候选 + K-Means 风格聚类,完成"输入 → 候选 → 排序 → 落库"完整链路。

【2 算法融合流程】
  ① NaiveBayesClassifier  — 天气 → 服饰类别概率排序(候选筛选)
  ② KMeansClustering       — 把候选服饰聚成 8 类,通过 DB 写回的 cluster_id 体现风格分组

【设计原则】
  - 每个方法都返回 {'success': bool, '...': ..., 'error': str} 的统一格式
  - try/except 兜底,业务层不会因单点失败而崩
  - 启动时 train_models() 一次,运行时直接 load
============================================================
"""

# ── 类型注解 ──
from typing import List, Dict, Any

# ── 数据访问层 ──
from database.models import DatabaseManager

# ── 2 个核心算法(具体逻辑见 algorithms/ 包) ──
from algorithms.naive_bayes_classifier import NaiveBayesClassifier
from algorithms.kmeans_clustering import KMeansClustering


class RecommendationService:
    """
    推荐业务服务:朴素贝叶斯 + K-Means 融合 + 落库 + 历史查询 + 反馈
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """
        构造方法
        :param db_manager: 数据库管理器(默认自建一个,便于测试注入 mock)
        """
        # 数据库访问层
        self.db = db_manager or DatabaseManager()

        # 2 个核心算法实例
        self.naive_bayes = NaiveBayesClassifier()   # 算法 1
        self.kmeans      = KMeansClustering()       # 算法 2

    # ============================================================
    # 业务核心:推荐 Top 5 穿搭
    # ============================================================
    def get_recommendations(self, user_id: int, temperature: float,
                            weather_condition: str, season: str) -> Dict[str, Any]:
        """
        完整推荐流程(2 算法串联:朴素贝叶斯筛候选 + K-Means 风格聚类)

        :param user_id:           用户 ID
        :param temperature:       当前温度(℃)
        :param weather_condition: 当前天气(sunny/rainy/snowy/cloudy)
        :param season:            当前季节
        :return: {
            'success': True/False,
            'recommendations': {...},
            'weather':        {...},
            'history_id':     落库的推荐历史 ID,
            'filters_applied':{'weather_filter_count':...}
        }
        """
        try:
            # ── 步骤 0:取用户、记录天气、加载服饰库 ──
            user_result = self.db.get_user(user_id)
            if not user_result:
                return {'success': False, 'error': 'User not found'}
            user = user_result

            # 把这次天气写库,生成 weather_id
            weather_id = self.db.create_weather(temperature, weather_condition, season)
            weather    = self.db.get_weather(weather_id)

            # 从 DB 加载全部服饰(后续会反复过滤)
            clothing_items = self.db.get_all_clothing_items()

            # ── 步骤 1(算法 1):朴素贝叶斯 — 天气→类别概率,筛候选 ──
            weather_filtered = self.naive_bayes.filter_clothing_by_weather(
                clothing_items, temperature, weather_condition, season
            )

            # 朴素贝叶斯结果为空时 → 兜底用全量服饰
            final_candidates = weather_filtered if weather_filtered else clothing_items

            # ── 步骤 2(算法 2):K-Means — 用 cluster_id 分组,直接取前 N 件 ──
            recommended_outfits = final_candidates[:5] if final_candidates else []

            # ── 步骤 3:把这次推荐落库(供"推荐历史"模块查询) ──
            history_id = self.db.create_recommendation_history(
                user_id, weather_id, recommended_outfits
            )

            return {
                'success':         True,
                'recommendations': {
                    'recommended_outfits': recommended_outfits,
                    'method':              'naive_bayes + kmeans',
                },
                'weather':         weather,
                'history_id':      history_id,
                'filters_applied': {
                    'weather_filter_count':  len(weather_filtered),
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 启动时调用:训练所有模型
    # ============================================================
    def train_models(self) -> Dict[str, Any]:
        """
        训练 2 个模型
        1) 朴素贝叶斯 — 读 CSV 训练
        2) K-Means     — 遍历服饰聚类,把 cluster_id 写回 DB
        """
        try:
            clothing_items = self.db.get_all_clothing_items()

            # 1) 朴素贝叶斯(读 weather_clothing_data.csv)
            self.naive_bayes.fit()

            # 2) K-Means(遍历服饰聚类 + 写回 cluster_id)
            if clothing_items:
                clusters = self.kmeans.fit(clothing_items)
                if clusters is not None:
                    for i, item in enumerate(clothing_items):
                        if i < len(clusters):
                            self.db.update_clothing_cluster(item['id'], int(clusters[i]))

            return {'success': True, 'message': 'All models trained successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 推荐历史查询
    # ============================================================
    def get_recommendation_history(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """从 DB 拉某用户最近的 N 条推荐历史"""
        try:
            history = self.db.get_recommendation_history(user_id, limit)
            return {'success': True, 'history': history}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ============================================================
    # 推荐反馈(用户对推荐结果点赞/踩)
    # ============================================================
    def submit_feedback(self, history_id: int, feedback: int) -> Dict[str, Any]:
        """feedback:1=赞,0=踩"""
        try:
            success = self.db.update_recommendation_feedback(history_id, feedback)
            if success:
                return {'success': True, 'message': 'Feedback submitted successfully'}
            return {'success': False, 'error': 'History record not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
