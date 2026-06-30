"""
============================================================
算法 2:K-Means 聚类(KMeansClustering)
============================================================
【业务定位】穿搭推荐流程的第 ③ 步
  输入:服饰字典列表(每件含 image_path、category、color、style、season 等)
  输出:每件服饰归属的聚类 ID(0~7,共 8 类风格)
  用途:把候选服饰按"风格"自动分组,后续可在指定聚类内推荐,保证风格统一。

【算法原理】sklearn KMeans
  - 无监督聚类,无需标注数据
  - 先随机选 8 个中心点,迭代"分配-更新"直到中心点稳定
  - 这里把"图片颜色直方图 + 文字元数据"拼接成 128 维特征向量

【为什么是 8 类?】
  与项目业务定义的 8 大风格(casual/elegant/sporty/business/street/sweet/
  japanese/korean)一一对应,在 Config.KMEANS_CLUSTERS 中配置。
============================================================
"""

# ── 标准库与第三方依赖 ──
import os                       # 路径判断
import joblib                   # 模型持久化
import numpy as np              # 数值计算
from PIL import Image           # 图像读取
from sklearn.cluster import KMeans                                       # K-Means 聚类
from sklearn.preprocessing import StandardScaler                         # 特征标准化(让各维度量纲一致)
from typing import List, Dict, Any, Tuple
from config import Config         # 配置(KMEANS_CLUSTERS=8、模型目录)


class KMeansClustering:
    """
    K-Means 聚类:把服饰按"图像特征 + 元数据"聚成 8 大风格簇
    """

    def __init__(self, n_clusters: int = None, model_path: str = None):
        """
        构造方法
        :param n_clusters: 聚类簇数,默认从配置读(8)
        :param model_path: 模型保存路径
        """
        # 聚类簇数:8 大风格
        self.n_clusters = n_clusters or Config.KMEANS_CLUSTERS

        # 模型文件路径(.pkl)
        self.model_path = model_path or os.path.join(Config.MODELS_DIR, 'kmeans_model.pkl')
        self.scaler_path = os.path.join(Config.MODELS_DIR, 'scaler.pkl')   # 标准化器单独保存

        # sklearn KMeans 对象(在 fit() 中实例化)
        self.model = None
        # 标准化器:把不同量纲的特征拉到同一尺度(像素 0~255 vs 标签 0~5)
        self.scaler = None

        # 单件服饰的图像特征维度(颜色直方图 96 + 均值 3 + 标准差 3 + 边缘 16 + 凑齐 128)
        self.feature_dim = 128

    # ============================================================
    # 特征工程 1:从图片中提取"颜色直方图 + 边缘"特征
    # ============================================================
    def extract_image_features(self, image_path: str) -> np.ndarray:
        """
        把图片降采样到 64x64 → 拆 RGB 通道 → 统计直方图 + 边缘
        :param image_path: 图片绝对路径
        :return: 128 维 numpy 向量
        """
        try:
            # 1) 读图 + 缩到 64x64(降采样提速,丢弃细节)
            img = Image.open(image_path)
            img = img.resize((64, 64))
            img_array = np.array(img)

            # 2) 兼容灰度图(2 维)和 RGBA 图(4 通道)
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)         # 灰度 → 3 通道
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]                        # RGBA → RGB

            # 3) RGB 三通道各做 32-bin 直方图(每 bin 表示一个颜色区间)
            hist_r, _ = np.histogram(img_array[:, :, 0], bins=32, range=(0, 256))
            hist_g, _ = np.histogram(img_array[:, :, 1], bins=32, range=(0, 256))
            hist_b, _ = np.histogram(img_array[:, :, 2], bins=32, range=(0, 256))
            color_hist = np.concatenate([hist_r, hist_g, hist_b])      # 96 维颜色直方图

            # 4) 全图均值 + 标准差(反映"整体色调明暗")
            mean_color = np.mean(img_array, axis=(0, 1))               # 3 维
            std_color  = np.std(img_array, axis=(0, 1))                # 3 维

            # 5) 灰度图 + 边缘检测,统计 16-bin 直方图(反映"图案复杂度")
            gray_img = Image.fromarray(img_array).convert('L')
            edges = np.array(gray_img.filter(ImageFilter.FIND_EDGES))  # 边缘强度图
            edge_hist, _ = np.histogram(edges, bins=16, range=(0, 256))  # 16 维

            # 6) 拼接所有特征 + 归一化(直方图转为概率分布,均值标准差除以 255)
            features = np.concatenate([
                color_hist / np.sum(color_hist + 1e-6),   # 颜色概率分布
                mean_color / 255.0,                       # 0~1 归一化
                std_color  / 255.0,
                edge_hist  / np.sum(edge_hist + 1e-6)
            ])

            # 7) 截断/补 0 到固定 128 维(模型要求输入维度一致)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]

            return features
        except Exception as e:
            # 图片读不出(损坏/路径错)→ 返回全 0 向量,不影响整体聚类
            print(f"Error extracting features from {image_path}: {e}")
            return np.zeros(self.feature_dim)

    # ============================================================
    # 特征工程 2:从元数据提取"类别/颜色/风格/季节"的标签编码
    # ============================================================
    def extract_metadata_features(self, item: Dict[str, Any]) -> np.ndarray:
        """
        把服饰的"类别 + 颜色 + 风格 + 季节 + 适用温度"转成数值向量
        用于和图像特征拼接,弥补纯图像特征对"语义"不敏感的缺点。
        """
        features = []

        # 类别 → 数值(0~5)
        category_map = {'top': 0, 'bottom': 1, 'dress': 2, 'outerwear': 3, 'shoes': 4, 'accessory': 5}
        features.append(category_map.get(item.get('category', 'top'), 0) / 5.0)

        # 颜色 → 数值(0~7)
        color_map = {'red': 0, 'blue': 1, 'green': 2, 'black': 3, 'white': 4,
                     'gray': 5, 'yellow': 6, 'pink': 7}
        features.append(color_map.get(item.get('color', 'black'), 0) / 7.0)

        # 风格 → 数值(0~4)
        style_map = {'casual': 0, 'formal': 1, 'sporty': 2, 'elegant': 3, 'streetwear': 4}
        features.append(style_map.get(item.get('style', 'casual'), 0) / 4.0)

        # 季节 → 数值(0~4)
        season_map = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3, 'all': 4}
        features.append(season_map.get(item.get('season', 'all'), 0) / 4.0)

        # 温度区间(归一化到 0~1)
        # 兜底:数据库里字段为 NULL 时用合理默认值,避免 None + 10 报错
        temp_min = item.get('suitable_temperature_min', -10) or -10
        temp_max = item.get('suitable_temperature_max',  40) or  40
        features.extend([(temp_min + 10) / 50.0, (temp_max + 10) / 50.0])

        return np.array(features)

    # ============================================================
    # 训练:把所有服饰的特征向量丢进 KMeans
    # ============================================================
    def fit(self, clothing_items: List[Dict[str, Any]]):
        """
        用服饰列表训练 KMeans,并把每件衣服的聚类 ID 写回数据库
        :param clothing_items: 服饰字典列表
        :return: 每个服饰的聚类 ID 数组(供调用方写库)
        """
        features_list = []

        for item in clothing_items:
            # 1) 图像特征(无图则全 0)
            image_features = np.zeros(self.feature_dim)
            if item.get('image_path') and os.path.exists(item['image_path']):
                image_features = self.extract_image_features(item['image_path'])

            # 2) 元数据特征
            metadata_features = self.extract_metadata_features(item)

            # 3) 拼接(图像 128 维 + 元数据 6 维 = 134 维)
            combined_features = np.concatenate([image_features, metadata_features])
            features_list.append(combined_features)

        if not features_list:
            print("No items to cluster")
            return

        X = np.array(features_list)

        # 4) 标准化(每列变成均值 0 方差 1,避免图像像素值 0~255 主导距离计算)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 5) 实际聚类数 = min(配置簇数, 样本数),避免样本太少时崩溃
        n_samples = len(X_scaled)
        actual_clusters = min(self.n_clusters, n_samples)
        if actual_clusters < self.n_clusters:
            print(f"Adjusting clusters from {self.n_clusters} to {actual_clusters}")

        # 6) KMeans 训练
        self.model = KMeans(n_clusters=actual_clusters, random_state=Config.RANDOM_STATE)
        clusters = self.model.fit_predict(X_scaled)  # fit + 一次性预测每条样本的簇 ID

        # 7) 保存模型 + 标准化器(下次启动无需重训)
        self.save_model()
        print(f"K-Means model trained with {actual_clusters} clusters")

        return clusters

    # ============================================================
    # 预测:单件服饰 → 归属哪个簇
    # ============================================================
    def predict(self, item: Dict[str, Any]) -> int:
        """预测单件服饰的聚类 ID(用于新衣服入库时自动打标签)"""
        if self.model is None:
            self.load_model()

        # 同样的特征工程:图像 + 元数据
        image_features = np.zeros(self.feature_dim)
        if item.get('image_path') and os.path.exists(item['image_path']):
            image_features = self.extract_image_features(item['image_path'])
        metadata_features = self.extract_metadata_features(item)
        combined_features = np.concatenate([image_features, metadata_features])

        # 标准化(必须用训练时的 scaler,否则量纲不一致)
        X = np.array([combined_features])
        X_scaled = self.scaler.transform(X)

        cluster = self.model.predict(X_scaled)[0]
        return int(cluster)

    # ============================================================
    # 业务方法:从服饰列表中筛出"指定簇"的衣服
    # ============================================================
    def get_cluster_items(self, clothing_items: List[Dict[str, Any]],
                         cluster_id: int) -> List[Dict[str, Any]]:
        """根据聚类 ID 取出所有属于该簇的服饰(数据库里已经存了 cluster_id)"""
        if self.model is None:
            self.load_model()
        return [item for item in clothing_items if item.get('cluster_id') == cluster_id]

    # ============================================================
    # 业务方法:聚类 ID → 风格中文名
    # ============================================================
    def get_style_name(self, cluster_id: int) -> str:
        """
        8 个聚类簇的"风格命名",按簇 ID 循环分配
        (聚类结果是无序的,这里用 ID 索引对应一个固定的风格列表)
        """
        style_names = [
            'Classic Elegant',     # 经典优雅
            'Casual Streetwear',   # 休闲街头
            'Formal Business',     # 商务正式
            'Sporty Active',       # 运动活力
            'Bohemian Chic',       # 波西米亚
            'Minimalist Modern',   # 极简现代
            'Vintage Retro',       # 复古
            'Trendy Fashion'       # 潮流时尚
        ]
        return style_names[cluster_id % len(style_names)]

    # ============================================================
    # 持久化
    # ============================================================
    def save_model(self):
        """保存 KMeans 模型 + 标准化器(分别存为 .pkl)"""
        joblib.dump(self.model,  self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """从磁盘加载 KMeans + 标准化器;没有则提示未训练"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model  = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            print("No pre-trained K-Means model found")


# ── 兼容 PIL 旧版本:ImageFilter 可能在某些精简版 Pillow 中不存在 ──
try:
    from PIL import ImageFilter
except ImportError:
    # 占位类,避免旧环境 import 失败
    class ImageFilter:
        FIND_EDGES = None
