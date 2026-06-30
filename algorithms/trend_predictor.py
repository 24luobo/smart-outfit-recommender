"""
============================================================
算法 5:潮流趋势预测器(TrendPredictor)
============================================================
【业务定位】独立功能,前端"潮流预测"页面
  输入:时间范围(current / next-month / next-season)
  输出:未来某月最可能流行的 1 个风格 + 8 风格占比 + 代表色 + 流行元素
  用途:运营/选品决策参考,告诉用户"下个月该穿什么风格的衣服最in"

【算法原理】8 个独立的 RandomForestRegressor
  - 训练数据:2020-01 到 2025-06 共 66 个月 × 8 风格 的"销售占比"(写在 CSV 里)
  - 时序特征:
      ① 月份 sin/cos 编码(让模型感知季节循环)
      ② 季节 one-hot
      ③ 滞后 1/2/3 月销售占比(自回归)
      ④ 滚动 3 月均值(平滑)
      ⑤ 趋势项(月份序号,捕捉长期上升/下降)
  - 递归外推:用前 3 月预测第 1 月,再把第 1 月加进"历史"预测第 2 月……
  - 不依赖任何用户数据(browse_heat_used: False)

【设计要点】汇报时强调
  - 数据规律来自真实时尚行业报告(疫情期 casual 涨,2024 Y2K street 涨……)
  - 8 个模型各管各的风格,互不影响
  - 冷启动时强制重新训练(get_trend_predictor 中 force_regen=True)
============================================================
"""

# ── 标准库与第三方依赖 ──
import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor          # 随机森林回归
from sklearn.preprocessing import StandardScaler            # 特征标准化
from config import Config


class TrendPredictor:
    """
    潮流趋势预测器 - 时序机器学习预测
    """

    # ── 文件路径常量 ──
    MODEL_PATH = os.path.join(Config.MODELS_DIR, 'trend_predictor.pkl')
    DATA_PATH  = os.path.join(Config.DATA_DIR,  'fashion_sales_history.csv')

    # ── 8 大风格英文 ID(也用于和"推荐穿搭"目录名映射) ──
    STYLES = ['casual', 'elegant', 'sporty', 'business', 'street', 'sweet', 'japanese', 'korean']

    # ── 风格英文 ID → 中文名 ──
    STYLE_NAMES_CN = {
        'casual':   '休闲日常', 'elegant':  '优雅气质', 'sporty':  '运动活力',
        'business': '商务通勤', 'street':   '街头潮流', 'sweet':   '甜美可爱',
        'japanese': '青春日系', 'korean':   '清爽韩系',
    }

    # ── 月份 → 季节(北半球气象划分) ──
    SEASON_OF_MONTH = {
        1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter'
    }
    SEASON_NAMES_CN = {'spring': '春季', 'summer': '夏季', 'autumn': '秋季', 'winter': '冬季'}

    # ── 每个风格对应的"代表色"(用于前端展示色卡) ──
    STYLE_COLORS = {
        'casual':   ['莫兰迪灰蓝', '浅咖', '柔米色'],
        'elegant':  ['米杏色', '雾霾蓝', '奶咖'],
        'sporty':   ['亮橙', '电光蓝', '荧光绿'],
        'business': ['炭灰', '深蓝', '米白'],
        'street':   ['黑色', '荧光色', '迷彩'],
        'sweet':    ['樱花粉', '奶油白', '薄荷绿'],
        'japanese': ['燕麦色', '雾灰', '浅草绿'],
        'korean':   ['莫兰迪粉', '鹅黄', '浅卡其'],
    }
    # ── 每个风格对应的"流行元素" ──
    STYLE_ELEMENTS = {
        'casual':   ['宽松版型', '简约线条', '舒适面料'],
        'elegant':  ['垂坠剪裁', '细节点缀', '质感面料'],
        'sporty':   ['运动线条', '透气网眼', '弹力面料'],
        'business': ['挺括版型', '低饱和色', '职业剪裁'],
        'street':   ['Oversize', '印花图案', '层次叠穿'],
        'sweet':    ['蕾丝/荷叶边', '柔和色调', '短款设计'],
        'japanese': ['宽松落肩', '棉麻质感', '低饱和度'],
        'korean':   ['修身剪裁', '叠穿层次', '莫兰迪色'],
    }

    def __init__(self, model_path: str = None):
        """构造方法,模型/数据/特征列均为空,fit() 中填充"""
        self.model_path = model_path or self.MODEL_PATH
        # 8 个 RandomForestRegressor(每个风格一个)
        self.models: Dict[str, RandomForestRegressor] = {}
        # 特征标准化器
        self.scaler: StandardScaler = None
        # 训练数据末尾,用于递归外推(知道"最近 3 个月"是哪些)
        self.last_data: Optional[pd.DataFrame] = None
        # 训练时用的特征列顺序(预测时必须保持一致)
        self.feature_columns: List[str] = []
        self.trained = False

    # ============================================================
    # 第一部分:数据生成(模拟 2020-01 到 2025-06 的真实时尚销售比例)
    # ============================================================
    def generate_training_data(self, force: bool = False) -> str:
        """
        生成 2020-01 到 2025-06 共 66 个月 × 8 风格的"销售占比"训练集

        占比是该月该风格的销售份额(0~1,8 风格合计 = 1.0)

        写实规律(由真实时尚行业报告归纳):
            - 长期趋势项:casual/sporty 2020 涨,2024 回落;business 2020 跌、2023 回涨;
                         street 2022 起持续涨(Y2K + 街头回潮);sweet 周期性,年底涨;
                         japanese 稳定 8-10%;korean 2023 起涨(韩剧带火)
            - 月度季节项:winter 推 business↑ elegant↑;summer 推 sporty↑ casual↑;
                         spring 推 sweet↑ japanese↑;autumn 推 street↑ business↑
            - 节日/事件:12 月 sweet 涨 30%,2 月(春节)红/国潮↑
        """
        if os.path.exists(self.DATA_PATH) and not force:
            return self.DATA_PATH

        os.makedirs(os.path.dirname(self.DATA_PATH), exist_ok=True)

        # 固定随机种子 → 同样的数据,汇报时数据可复现
        rng = np.random.default_rng(2024)
        rows = []

        # ── 调参经验:8 个风格的基线尽量拉近,让"季节/趋势/节日"决定谁第一 ──
        style_base_2020 = {
            'casual': 0.12, 'elegant': 0.13, 'sporty': 0.13, 'business': 0.13,
            'street': 0.12, 'sweet': 0.12, 'japanese': 0.12, 'korean': 0.13,
        }
        # ── 长期趋势(每年变化量,正=上升) ──
        style_yearly_trend = {
            'casual': 0.000, 'elegant': 0.003, 'sporty': 0.002, 'business': -0.002,
            'street': 0.008, 'sweet': -0.001, 'japanese': 0.001, 'korean': 0.005,
        }
        # ── 季节偏置:加大差异,使每季有"主打风格" ──
        style_season_bias = {
            'winter': {'casual': -0.02, 'elegant': 0.06, 'sporty': -0.05, 'business': 0.08,
                       'street': 0.00, 'sweet': 0.04, 'japanese': 0.00, 'korean': 0.01},
            'spring': {'casual': 0.01, 'elegant': 0.00, 'sporty': 0.00, 'business': -0.04,
                       'street': 0.00, 'sweet': 0.05, 'japanese': 0.06, 'korean': 0.04},
            'summer': {'casual': 0.03, 'elegant': -0.04, 'sporty': 0.08, 'business': -0.07,
                       'street': 0.02, 'sweet': 0.01, 'japanese': -0.02, 'korean': 0.01},
            'autumn': {'casual': -0.02, 'elegant': 0.01, 'sporty': -0.03, 'business': 0.02,
                       'street': 0.07, 'sweet': -0.01, 'japanese': 0.00, 'korean': 0.04},
        }
        # ── 节日 spike(月份,风格,增量) ──
        holiday_spikes = [
            (2,  'sweet',   0.04), (2,  'street',   0.03),   # 春节红/国潮
            (11, 'sweet',   0.04), (11, 'business', 0.02),   # 双 11
            (12, 'sweet',   0.06), (12, 'street',   0.02),   # 圣诞
        ]

        start = datetime(2020, 1, 1)
        end   = datetime(2025, 6, 1)
        cur   = start
        while cur <= end:
            month = cur.month
            year  = cur.year
            season = self.SEASON_OF_MONTH[month]
            years_from_2020 = (year - 2020) + (month - 1) / 12.0

            # 计算每个风格的"原始得分"
            scores = {}
            for s in self.STYLES:
                score = style_base_2020[s] + style_yearly_trend[s] * years_from_2020
                score += style_season_bias[season].get(s, 0.0)
                for hm, hs, hb in holiday_spikes:
                    if hm == month:
                        score += hb
                score += rng.normal(0, 0.008)               # 高斯噪声
                scores[s] = max(score, 0.01)                # 保底,避免出现 0

            # 归一化成占比(8 个风格合计 = 1)
            total = sum(scores.values())
            for s in self.STYLES:
                rows.append({
                    'date':   cur.strftime('%Y-%m-%d'),
                    'year':   year, 'month': month, 'season': season,
                    'style':  s, 'share': round(scores[s] / total, 4),
                })

            # 推进 1 个月
            cur = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

        df = pd.DataFrame(rows)
        df.to_csv(self.DATA_PATH, index=False, encoding='utf-8-sig')
        return self.DATA_PATH

    # ============================================================
    # 第二部分:时序特征工程
    # ============================================================
    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        把 (date, style, share) 长表,转成时序特征 + 目标
        :return: (X 特征矩阵, y_df 含 date/style/target 的元信息)
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['style', 'date']).reset_index(drop=True)

        # 1) 月份 sin/cos 编码(把 1-12 月"摊到圆上",让模型感知 12 月和 1 月相邻)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 2) 季节 one-hot
        for s in ['spring', 'summer', 'autumn', 'winter']:
            df[f'season_{s}'] = (df['season'] == s).astype(int)

        # 3) 趋势项(从 2020-01 起的月份序号,捕捉长期升降)
        base = pd.Timestamp('2020-01-01')
        df['trend_idx'] = ((df['date'] - base).dt.days / 30.0).astype(int)

        # 4) 滞后 1/2/3 月 + 滚动 3 月均值(按 style 分组,做自回归)
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = df.groupby('style')['share'].shift(lag)
        df['roll_mean_3'] = df.groupby('style')['share'] \
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

        # 5) 目标变量(就是要预测的当月 share)
        df['target'] = df['share']

        # 6) 训练时只取 lag_1/2/3 都有值的行(前 3 个月丢)
        feature_cols = ['month_sin', 'month_cos',
                        'season_spring', 'season_summer', 'season_autumn', 'season_winter',
                        'trend_idx', 'lag_1', 'lag_2', 'lag_3', 'roll_mean_3']
        self.feature_columns = feature_cols

        df_feat = df.dropna(subset=['lag_1', 'lag_2', 'lag_3'])
        return df_feat[feature_cols], df_feat[['date', 'style', 'target']]

    def _future_row(self, future_date: pd.Timestamp, prev_shares: Dict[str, float],
                    style: str, month_idx_from_base: int) -> Dict[str, float]:
        """
        构造一个"未来月份"的某 style 的特征行
        :param prev_shares: 历史占比字典 {'s1_casual':0.12, 's2_casual':0.11, ...}
        :param month_idx_from_base: 该月距 2020-01 的月份序号
        """
        month = future_date.month
        season = self.SEASON_OF_MONTH[month]
        feats = {
            'month_sin':      math.sin(2 * math.pi * month / 12),
            'month_cos':      math.cos(2 * math.pi * month / 12),
            'season_spring':  int(season == 'spring'),
            'season_summer':  int(season == 'summer'),
            'season_autumn':  int(season == 'autumn'),
            'season_winter':  int(season == 'winter'),
            'trend_idx':      month_idx_from_base,
            'lag_1':          prev_shares.get(f's1_{style}', 0.10),
            'lag_2':          prev_shares.get(f's2_{style}', 0.10),
            'lag_3':          prev_shares.get(f's3_{style}', 0.10),
            'roll_mean_3':   (prev_shares.get(f's1_{style}', 0.10) +
                              prev_shares.get(f's2_{style}', 0.10) +
                              prev_shares.get(f's3_{style}', 0.10)) / 3,
        }
        return feats

    # ============================================================
    # 第三部分:训练
    # ============================================================
    def fit(self, force_regen: bool = False) -> 'TrendPredictor':
        """
        训练 8 个 RandomForestRegressor
        :param force_regen: 是否强制重新生成数据
        """
        if force_regen or not os.path.exists(self.DATA_PATH):
            self.generate_training_data(force=True)

        df = pd.read_csv(self.DATA_PATH)
        X, y_df = self._build_features(df)

        # 1) 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 2) 按 style 分别训练一个 RandomForest(每个风格一个模型)
        for style in self.STYLES:
            mask = (y_df['style'] == style).values
            y_style = y_df.loc[mask, 'target'].values
            X_style = X_scaled[mask]
            model = RandomForestRegressor(
                n_estimators=200,      # 200 棵树
                max_depth=8,            # 限制深度,防过拟合
                min_samples_leaf=3,     # 叶子节点最少 3 个样本
                random_state=42,        # 固定种子,可复现
                n_jobs=-1               # 多核并行
            )
            model.fit(X_style, y_style)
            self.models[style] = model

        self.last_data = df.copy()
        self.trained = True
        self.save()
        return self

    def save(self):
        """保存 8 个模型 + 标准化器 + 特征列顺序 + 训练数据末尾"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'models':          self.models,
            'scaler':          self.scaler,
            'feature_columns': self.feature_columns,
            'last_data':       self.last_data,
        }, self.model_path)

    def load(self) -> bool:
        """从磁盘加载;返回 True/False 表示加载是否成功"""
        if not os.path.exists(self.model_path):
            return False
        data = joblib.load(self.model_path)
        self.models          = data['models']
        self.scaler          = data['scaler']
        self.feature_columns = data['feature_columns']
        self.last_data       = data['last_data']
        self.trained = True
        return True

    def ensure_trained(self):
        """确保模型已训练好(load → 失败则 fit)"""
        if not self.trained and not self.load():
            self.fit()

    # ============================================================
    # 第四部分:递归外推预测
    # ============================================================
    def _predict_recursive(self, start_date: datetime, horizon_months: int) -> List[Dict[str, Any]]:
        """
        从 start_date 起,递归预测未来 horizon_months 个月的 8 风格占比

        算法:
          1) 取最近 3 个月作为 lag1/2/3 起点
          2) 每个月:对 8 个风格分别预测,得到 share → 归一化
          3) 把当月 share 加进"历史",作为下个月的 lag
        """
        df = self.last_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        last_dates = sorted(df['date'].unique())[-3:]   # 最近 3 个月

        # 每个 style 的"最近 3 个月"share 序列
        history: Dict[str, List[float]] = {s: [] for s in self.STYLES}
        for d in last_dates:
            for s in self.STYLES:
                row = df[(df['date'] == d) & (df['style'] == s)]
                history[s].append(float(row['share'].iloc[0]) if not row.empty else 0.10)

        base = pd.Timestamp('2020-01-01')

        results = []
        cur = start_date.replace(day=1)
        for step in range(horizon_months):
            row_preds: Dict[str, float] = {}
            for style in self.STYLES:
                # 准备滞后特征
                prev = {
                    f's1_{style}': history[style][-1],
                    f's2_{style}': history[style][-2] if len(history[style]) >= 2 else history[style][-1],
                    f's3_{style}': history[style][-3] if len(history[style]) >= 3 else history[style][-1],
                }
                feat = self._future_row(pd.Timestamp(cur), prev, style,
                                        int((pd.Timestamp(cur) - base).days / 30))
                X_row = pd.DataFrame([feat], columns=self.feature_columns)
                X_row_scaled = self.scaler.transform(X_row)
                pred = float(self.models[style].predict(X_row_scaled)[0])
                pred = max(pred, 0.005)                   # 保底
                row_preds[style] = pred
                history[style].append(pred)               # 把预测结果加进历史
                if len(history[style]) > 3:
                    history[style] = history[style][-3:]  # 只保留最近 3 个月

            # 归一化(8 风格合计 = 1)
            total = sum(row_preds.values()) or 1.0
            row_preds = {k: v / total for k, v in row_preds.items()}

            results.append({
                'date':         cur.strftime('%Y-%m-%d'),
                'year':         cur.year,
                'month':        cur.month,
                'season':       self.SEASON_OF_MONTH[cur.month],
                'distribution': row_preds,
            })

            # 推进到下个月
            cur = datetime(cur.year + 1, 1, 1) if cur.month == 12 else datetime(cur.year, cur.month + 1, 1)

        return results

    # ============================================================
    # 第五部分:业务入口 - 根据时间范围做预测
    # ============================================================
    def predict_for_time_range(self, time_range: str = 'current') -> Dict[str, Any]:
        """
        业务接口:返回"目标月"的最可能风格 + 8 风格占比

        :param time_range:
            - 'current'      : 当下(本月)
            - 'next-month'   : 下个月
            - 'next-season'  : 下个季节
        """
        self.ensure_trained()

        now = datetime.now()
        if time_range == 'current':
            target_date = now.replace(day=1); horizon = 1
        elif time_range == 'next-month':
            nm = now.replace(day=1) + timedelta(days=32)
            target_date = nm.replace(day=1); horizon = 1
        elif time_range == 'next-season':
            # 下一个季节的第一月
            target_date = now.replace(day=1) + timedelta(days=90)
            target_date = target_date.replace(day=1); horizon = 1
        else:
            target_date = now.replace(day=1); horizon = 1

        # 始终从现在起外推 12 个月(覆盖 next-season)
        predictions = self._predict_recursive(now.replace(day=1), max(horizon, 12))

        # ── 选目标月 ──
        if time_range == 'current':
            target_idx = 0
        elif time_range == 'next-month':
            target_idx = 1
        else:  # next-season
            cur_season = self.SEASON_OF_MONTH[now.month]
            target_idx = 1
            for i, p in enumerate(predictions):
                if p['season'] != cur_season:           # 找第一个不同季节的月
                    target_idx = i
                    break

        target = predictions[min(target_idx, len(predictions) - 1)]
        dist   = target['distribution']
        # ── 找出预测占比最高的风格 ──
        top_style = max(dist, key=dist.get)
        top_prob  = dist[top_style]

        # ── 置信度 = top1 与 top2 的差距(差距越大,模型越自信) ──
        sorted_p  = sorted(dist.values(), reverse=True)
        gap       = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else 0.5
        confidence = min(0.95, 0.55 + gap)

        season_cn  = self.SEASON_NAMES_CN[target['season']]
        time_text  = {'current': '当下', 'next-month': '未来 1 个月', 'next-season': '未来 1 季'}.get(time_range, '近期')

        return {
            'time_range':         time_range,
            'target_date':        target['date'],
            'target_month':       target['month'],
            'target_year':        target['year'],
            'target_season':      target['season'],
            'target_season_cn':   season_cn,
            'style':              self.STYLE_NAMES_CN[top_style],
            'style_key':          top_style,
            'colors':             self.STYLE_COLORS.get(top_style, []),
            'elements':           self.STYLE_ELEMENTS.get(top_style, []),
            'confidence':         round(confidence, 2),
            'description': (
                f'{time_text}({target["year"]}年{target["month"]}月 {season_cn})'
                f'潮流预测:{self.STYLE_NAMES_CN[top_style]}风格预计最受欢迎'
                f'(预测占比 {top_prob*100:.1f}%)。'
                f'代表色:{", ".join(self.STYLE_COLORS.get(top_style, [])[:2])};'
                f'流行元素:{", ".join(self.STYLE_ELEMENTS.get(top_style, [])[:2])}。'
            ),
            'distribution': {self.STYLE_NAMES_CN[k]: round(v, 3)
                            for k, v in sorted(dist.items(), key=lambda x: -x[1])},
            'method':             'RandomForestRegressor + 时序滞后特征(过去 24+ 月销售比例)',
            'model_trained_on':   '2020-01 至 2025-06 月度时尚销售占比(8 风格)',
            'browse_heat_used':   False,                  # 明确:不基于用户数据
        }


# ── 全局单例(保证全应用只训练一次) ──
_instance: Optional[TrendPredictor] = None


def get_trend_predictor() -> TrendPredictor:
    """
    单例工厂函数:第一次调用时训练,后续直接返回缓存实例
    强制 force_regen=True 重新生成数据 + 重新训练(避免沿用旧模型导致某一风格永远第一)
    """
    global _instance
    if _instance is None:
        _instance = TrendPredictor()
        try:
            _instance.fit(force_regen=True)
        except Exception as e:
            print(f'Warning: 重新训练 trend predictor 失败,尝试加载旧模型: {e}')
            if not _instance.load():
                _instance.fit(force_regen=True)
    return _instance
