"""
============================================================
算法 1:朴素贝叶斯分类器(NaiveBayesClassifier)
============================================================
【业务定位】穿搭推荐流程的第 ① 步
  输入:温度(数值) + 天气状况(sunny/rainy/snowy/cloudy) + 季节
  输出:各类服饰(top/bottom/outerwear/dress/shoes)的概率分布
  用途:从全量服饰中筛出"最匹配当前天气"的类别,作为后续算法的候选集

【算法原理】sklearn CategoricalNB(适合离散特征)
  朴素贝叶斯公式:P(类别|特征) ∝ P(特征|类别) × P(类别)
  "朴素"指假设各特征之间相互独立,在小数据上效果稳定、可解释性强。

【训练数据】data/weather_clothing_data.csv
  若文件不存在,自动调用 _create_sample_data() 生成 500 条带规则的样本。
============================================================
"""

# ── 标准库与第三方依赖 ──
import os                                  # 路径拼接、目录创建
import joblib                              # 模型序列化(替代 pickle,支持大 numpy 数组)
import numpy as np                         # 数值计算
import pandas as pd                        # 表格数据加载
from sklearn.naive_bayes import CategoricalNB            # 朴素贝叶斯(分类特征)
from sklearn.preprocessing import LabelEncoder           # 字符串 → 数字编码
from typing import List, Dict, Any, Tuple                # 类型注解
from config import Config                                # 项目配置(数据/模型目录)


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器:根据"温度区间 + 天气 + 季节"预测服饰类别概率
    """

    def __init__(self, model_path: str = None):
        """
        构造方法
        :param model_path: 模型文件路径,默认保存到 models/naive_bayes_model.pkl
        """
        # 模型文件保存路径(pkl 格式,joblib 序列化)
        self.model_path = model_path or os.path.join(Config.MODELS_DIR, 'naive_bayes_model.pkl')

        # 字符串 → 整数的编码器字典,key 是特征列名
        # 朴素贝叶斯只能处理数值,字符串必须先编码
        self.label_encoders = {}

        # sklearn 朴素贝叶斯模型对象(在 fit() 中实例化)
        self.model = None

        # 输入特征列:温度区间、天气、季节(都是离散值)
        self.feature_columns = ['temperature_bin', 'weather_condition', 'season']

        # 目标列:服饰类别(top/bottom/outerwear/dress/shoes 等)
        self.target_column = 'category'

    # ------------------------------------------------------------------
    # 内部工具:温度 → 区间字符串(把连续值离散化)
    # ------------------------------------------------------------------
    def _bin_temperature(self, temp: float) -> str:
        """
        把连续温度值分成 4 个区间(冷/凉/暖/热)
        区间划分规则:
            < 10  → cold       (冷,适合外套/厚上衣)
            10~20 → cool       (凉,适合薄外套)
            20~28 → warm       (暖,适合短袖/裙)
            ≥ 28  → hot        (热,适合连衣裙/短裤)
        """
        if temp < 10:
            return 'cold'
        elif 10 <= temp < 20:
            return 'cool'
        elif 20 <= temp < 28:
            return 'warm'
        else:
            return 'hot'

    # ------------------------------------------------------------------
    # 内部工具:把 3 个原始输入组装成 1 行 DataFrame
    # ------------------------------------------------------------------
    def _prepare_features(self, temperature: float, weather_condition: str, season: str) -> pd.DataFrame:
        """
        特征工程:把用户输入的温度(数值)+ 天气(字符串)+ 季节(字符串)
                 转换为模型接受的 DataFrame 格式
        """
        temp_bin = self._bin_temperature(temperature)             # 温度数值 → 区间字符串
        data = {
            'temperature_bin':   [temp_bin],
            'weather_condition': [weather_condition],
            'season':            [season]
        }
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # 训练方法:从 CSV 读数据 → 编码 → 训练 → 保存
    # ------------------------------------------------------------------
    def fit(self, data_path: str = None):
        """
        训练朴素贝叶斯模型
        :param data_path: 训练 CSV 路径,默认 data/weather_clothing_data.csv
        """
        data_path = data_path or os.path.join(Config.DATA_DIR, 'weather_clothing_data.csv')

        # 数据文件不存在时,自动生成 500 条带规则标注的样本(便于冷启动)
        if not os.path.exists(data_path):
            self._create_sample_data(data_path)

        # 1) 读 CSV
        df = pd.read_csv(data_path)

        # 2) 拆分 X(特征) / y(标签)
        X = df[self.feature_columns].copy()
        y = df[self.target_column]

        # 3) 对每列做 LabelEncoder(把字符串映射成整数,朴素贝叶斯只认数字)
        for col in self.feature_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])     # fit + transform 一次完成
            self.label_encoders[col] = le         # 保存编码器,预测时要复用

        # 4) 标签也编码
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target

        # 5) 实例化并训练模型
        self.model = CategoricalNB()
        self.model.fit(X, y_encoded)

        # 6) 持久化(下次启动直接 load,无需重训)
        self.save_model()
        print("Naive Bayes model trained and saved.")

    # ------------------------------------------------------------------
    # 冷启动:无数据文件时,按"天气常识"生成 500 条带规则标注的样本
    # ------------------------------------------------------------------
    def _create_sample_data(self, data_path: str):
        """
        生成示例训练数据(规则:温度低 → 推外套,温度高 → 推裙装,等等)
        真实部署时可替换为人工标注的运营数据。
        注:category 用中文,与 clothing_items.category 一致
        """
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        data = []
        weather_conditions = ['sunny', 'rainy', 'snowy', 'cloudy']    # 4 种天气
        seasons            = ['spring', 'summer', 'autumn', 'winter'] # 4 个季节

        # 生成 500 条样本,温度均匀分布在 -5~35℃
        for _ in range(500):
            temp    = np.random.uniform(-5, 35)
            weather = np.random.choice(weather_conditions)
            season  = np.random.choice(seasons)

            # ── 关键规则:不同温度段推荐不同类别(中文) ──
            if temp < 10 or season == 'winter' or weather == 'snowy':
                # 冷天/冬季/下雪:推外套 + 上衣(厚款)
                category = np.random.choice(['外套', '上衣'], p=[0.4, 0.6])
            elif 10 <= temp < 20:
                # 微凉:上衣/下装/外套概率接近
                category = np.random.choice(['上衣', '下装', '外套'], p=[0.35, 0.35, 0.3])
            elif 20 <= temp < 28:
                # 温暖:连衣裙占比最高
                category = np.random.choice(['上衣', '下装', '连衣裙'], p=[0.3, 0.3, 0.4])
            else:
                # 热天:连衣裙更突出
                category = np.random.choice(['上衣', '连衣裙', '下装'], p=[0.25, 0.45, 0.3])

            data.append({
                'temperature_bin':   self._bin_temperature(temp),
                'weather_condition': weather,
                'season':            season,
                'category':          category
            })

        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False, encoding='utf-8-sig')
        print(f"Sample data created at {data_path}")

    # ------------------------------------------------------------------
    # 预测:输入温度/天气/季节 → 输出各类别概率并按概率降序返回
    # ------------------------------------------------------------------
    def predict(self, temperature: float, weather_condition: str, season: str) -> List[Dict[str, Any]]:
        """
        预测各类别概率
        :return: 形如 [{'category':'dress','probability':0.42}, ...] 的列表
        """
        # 模型未加载 → 优先从磁盘加载,没有就现场训练
        if self.model is None:
            self.load_model()

        # 1) 组装特征行
        X = self._prepare_features(temperature, weather_condition, season)

        # 2) 用训练时保存的编码器把字符串转回数字(必须用 transform 不用 fit_transform)
        for col in self.feature_columns:
            X[col] = self.label_encoders[col].transform(X[col])

        # 3) predict_proba 返回每个类别的概率(二维数组,取第 0 行)
        probabilities = self.model.predict_proba(X)[0]

        # 4) 把数字标签还原回字符串类别名
        categories = self.label_encoders['target'].inverse_transform(np.arange(len(probabilities)))

        # 5) 组装 (类别, 概率) 元组列表
        results = []
        for cat, prob in zip(categories, probabilities):
            results.append({
                'category':    cat,
                'probability': float(prob)
            })

        # 6) 按概率从高到低排序(便于业务层取 top-K)
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results

    # ------------------------------------------------------------------
    # 业务方法:用朴素贝叶斯筛选符合当前天气的服饰列表
    # ------------------------------------------------------------------
    def filter_clothing_by_weather(self, clothing_items: List[Dict[str, Any]],
                                    temperature: float, weather_condition: str,
                                    season: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        业务级过滤:从服饰库中挑出既"类别匹配天气概率 top-k"又"温度/季节合适"的衣服

        :param clothing_items:   服饰字典列表
        :param temperature:      当前温度(℃)
        :param weather_condition:当前天气
        :param season:           当前季节
        :param top_k:            选取概率最高的前 K 个类别
        :return: 过滤后的服饰列表(带 weather_match_score 字段)
        """
        # 1) 拿到各类别在当前天气下的概率
        predictions = self.predict(temperature, weather_condition, season)

        # 2) 选出 top-K 类别(例如:dress、top、bottom)
        suitable_categories = [p['category'] for p in predictions[:top_k]]

        # 3) 遍历服饰,做 3 重过滤:类别匹配 + 温度区间匹配 + 季节匹配
        filtered_items = []
        for item in clothing_items:
            # ── 过滤 1:类别必须在前 K 个高概率类别里 ──
            if item['category'] in suitable_categories:
                # ── 过滤 2:温度区间 ──
                temp_ok = True
                if item.get('suitable_temperature_min') is not None:
                    if temperature < item['suitable_temperature_min']:
                        temp_ok = False
                if item.get('suitable_temperature_max') is not None:
                    if temperature > item['suitable_temperature_max']:
                        temp_ok = False

                # ── 过滤 3:季节匹配(若衣服限定了季节且不是"全年") ──
                season_ok = True
                if item.get('season') and item['season'] != 'all':
                    if item['season'] != season:
                        season_ok = False

                if temp_ok and season_ok:
                    # 把"天气类别匹配概率"写到服饰字典上,供后续算法排序使用
                    category_prob = next((p['probability'] for p in predictions
                                         if p['category'] == item['category']), 0)
                    item['weather_match_score'] = category_prob
                    filtered_items.append(item)

        # 按匹配概率从高到低排序
        filtered_items.sort(key=lambda x: x.get('weather_match_score', 0), reverse=True)
        return filtered_items

    # ------------------------------------------------------------------
    # 持久化:把模型和编码器一起序列化到磁盘
    # ------------------------------------------------------------------
    def save_model(self):
        """保存模型到 .pkl 文件(下次启动直接 load,无需重训)"""
        data = {
            'model':          self.model,
            'label_encoders': self.label_encoders
        }
        joblib.dump(data, self.model_path)

    # ------------------------------------------------------------------
    # 加载:从磁盘恢复模型,没有就训练一份
    # ------------------------------------------------------------------
    def load_model(self):
        """从磁盘加载模型;模型文件不存在则现场训练"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model          = data['model']
            self.label_encoders = data['label_encoders']
        else:
            self.fit()
