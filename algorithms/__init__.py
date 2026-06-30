"""
============================================================
algorithms 包初始化文件
============================================================
作用:统一对外暴露 5 个机器学习算法类,方便业务层 `from algorithms import ...`
设计原则:这是 algorithms/ 包的"门面",所有具体实现类的导出都集中在这里,
        业务层无需关心算法文件的具体路径和类名,直接按"算法名"导入即可。
============================================================
"""

# ── 算法 1:朴素贝叶斯(天气 → 服饰类别概率分类)──────────────────
from .naive_bayes_classifier import NaiveBayesClassifier

# ── 算法 2:K-Means 聚类(服饰特征向量 → 8 类风格)───────────────
from .kmeans_clustering import KMeansClustering

# ── 算法 3:随机森林时序预测(过去 24+ 月销售占比 → 未来潮流)──
# TrendPredictor:核心类,get_trend_predictor:单例工厂函数(保证全应用只训练一次)
from .trend_predictor import TrendPredictor, get_trend_predictor


# ── `__all__` 列出本包对外公开的符号,`from algorithms import *` 时生效 ──
__all__ = [
    'NaiveBayesClassifier',          # 算法 1 类
    'KMeansClustering',              # 算法 2 类
    'TrendPredictor',                # 算法 3 类
    'get_trend_predictor',           # 算法 3 单例工厂
]
