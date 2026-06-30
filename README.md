# 智能穿搭推荐系统

基于机器学习的 Web 端智能穿搭推荐系统,集成 **5 个机器学习模块** + **多模型 LLM 助手**,为用户提供个性化穿搭方案、潮流预测、服饰评论生成。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-2.3%2B-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

---

## ✨ 核心亮点

- 🧠 **5 个机器学习算法** 协同工作
- 📈 **真时序潮流预测**(基于 2020-2025 历史销售数据 + RandomForest 回归,递归外推 12 个月)
- 💬 **AI 穿搭助手**(支持 DeepSeek / 智谱 GLM / 通义千问 / OpenAI / Ollama 5 个 LLM,无 key 自动降级本地规则)
- 🎨 **细粒度服饰评价生成**(材质 / 版型 / 风格 / 搭配 四维度,基于 6 大类目模板)
- 👗 **48 套实拍穿搭图库**(男/女 × 6 风格 × 4 季)
- ❤️ **完整用户功能**:新手引导 / 个人偏好 / 收藏 / 历史浏览 / 个人衣服 / 在线试衣
- 🤖 **零门槛游客体验**:刷新即弹引导,免登录可使用全部功能

---

## 🧠 3 个机器学习算法

| # | 算法 | 文件 | 作用 |
|---|------|------|------|
| 1 | **朴素贝叶斯** | `algorithms/naive_bayes_classifier.py` | 天气 → 服饰类别概率分类(CategoricalNB) |
| 2 | **K-Means 聚类** | `algorithms/kmeans_clustering.py` | 服饰特征向量聚类(8 类风格) |
| 3 | **随机森林时序预测** | `algorithms/trend_predictor.py` | 8 风格未来 1-12 月销售占比预测(RandomForestRegressor + 时序滞后特征) |

### 潮流预测算法细节
- **训练数据**:`data/fashion_sales_history.csv` — 2020-01 至 2025-06 共 66 个月 × 8 风格的销售占比(基于时尚行业报告归纳的写实规律)
- **特征工程**:月份 sin/cos 循环编码 + 季节 one-hot + 滞后 1/2/3 月 + 滚动 3 月均值 + 趋势项
- **模型**:每个风格单独训练一个 `RandomForestRegressor`(200 树,深度 8)
- **预测**:递归外推,最多预测未来 12 个月
- **不依赖任何用户数据**(`browse_heat_used: false`)

### 细粒度服饰评价算法
- **类目**:上衣 / 裤子 / 外套 / 连衣裙 / 裙子 / 配饰(6 大类)
- **维度**:材质 / 版型 / 风格 / 搭配(每类目 4-6 条模板)
- **生成逻辑**:基于服饰分类 + 风格标签从模板池中加权采样,确保不重复
- **位置**:`algorithms/clothing_comment_generator.py`

---

## 🛠️ 技术栈

- **后端**:Python 3.8+ / Flask 2.3+ / Flask-CORS
- **机器学习**:scikit-learn 1.3+(CategoricalNB / KMeans / RandomForest)
- **数据库**:SQLite(单文件,首次启动自动建表)
- **前端**:HTML5 + CSS3 + 原生 JavaScript(ES6+,无框架)
- **大模型**:DeepSeek / 智谱 GLM / 通义千问 / OpenAI / Ollama(走 OpenAI 兼容协议,纯 `urllib` 实现,无需 SDK)
- **数据可视化**:Pillow(图片处理)

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- pip

### 安装与启动(3 步)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. (可选)初始化示例数据 — 首次启动会自动建表
python scripts/init_sample_data.py

# 3. 启动 Flask
python app.py
```

浏览器打开:**http://127.0.0.1:5000/**

### 配置 AI 助手(可选)

复制 `.env.example` 为 `.env`,填入任一模型的 API Key:

```env
# 5 选 1
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key-here
```

未配置也能用,会自动降级到内置的 8 大场景规则回复(天气/约会/职场/小个子/显瘦/配色/男生/女生)。

### Windows 一键启动

```cmd
run.bat
```

---

## 📁 项目结构

```
lijie/
├── app.py                          # Flask 主入口 + 启动时训练所有 ML 模型
├── config.py                       # 配置(数据库路径、模型目录等)
├── requirements.txt                # Python 依赖
├── run.bat / run.sh                # 启动脚本
├── .env.example                    # AI 模型配置示例
│
├── algorithms/                     # 3 个机器学习算法
│   ├── naive_bayes_classifier.py          # 朴素贝叶斯(成员1)
│   ├── kmeans_clustering.py               # K-Means 聚类(成员2)
│   ├── trend_predictor.py                 # 随机森林时序预测(成员3)
│   └── clothing_comment_generator.py      # 细粒度服饰评价生成
│
├── services/                       # 业务逻辑层
│   ├── recommendation_service.py   # 朴素贝叶斯 + K-Means 融合推荐主流程
│   ├── clothing_service.py
│   ├── weather_service.py
│   ├── user_service.py
│   ├── favorite_service.py         # 收藏(去重 + 取消)
│   ├── browse_history_service.py   # 历史浏览 CRUD
│   └── llm_service.py              # 多模型 LLM 统一调用层
│
├── routes/                         # Flask 路由
│   ├── user_routes.py
│   ├── weather_routes.py
│   ├── clothing_routes.py
│   ├── recommendation_routes.py
│   ├── favorite_routes.py
│   ├── body_shape_routes.py
│   ├── auth_routes.py              # 游客/注册/登录
│   └── browse_history_routes.py
│
├── database/
│   ├── init_db.py                  # 8 张表自动初始化 + 幂等迁移
│   └── models.py                   # DatabaseManager 全量 CRUD
│
├── templates/                      # 13 个 HTML 页面
│   ├── base.html                   # 全局布局(fixed 顶/底导航 + 新手引导)
│   ├── index.html                  # 首页(智能穿搭推荐)
│   ├── clothing_browser.html       # 服饰库
│   ├── my_clothes.html             # 个人衣服
│   ├── recommendation.html         # 在线试衣
│   ├── trend_prediction.html       # 潮流预测(8 风格概率柱状图)
│   ├── ai_assistant.html           # AI 助手(5 模型切换)
│   ├── weather_input.html          # 天气输入
│   ├── body_shape.html             # 身材信息
│   ├── user_profile.html           # 我的(收藏/历史/偏好入口)
│   ├── favorites.html              # 收藏页
│   ├── history.html                # 历史浏览
│   └── preferences.html            # 个人偏好设置
│
├── static/
│   ├── css/ (style.css, auth.css, tryon.css)
│   ├── js/  (main.js, auth-simple.js, tryon.js)
│   └── assets/推荐穿搭/            # 48 个风格子目录(共 300+ 张实拍图)
│
├── data/                           # 训练数据
│   ├── weather_clothing_data.csv           # 朴素贝叶斯(天气→服饰类别)
│   ├── fashion_sales_history.csv           # 潮流预测(2020-2025 月度销售占比)
│   └── sample_clothing_items.csv           # 初始化样例
│
├── models/                         # 训练产物(运行时自动生成)
│   ├── naive_bayes_model.pkl
│   ├── kmeans_model.pkl
│   └── trend_predictor.pkl        # 8 个 RandomForest + StandardScaler
│
├── scripts/
│   └── init_sample_data.py        # 初始化数据库
│
└── tests/                          # 单元测试
```

---

## 🌐 页面与功能

| 路径 | 功能 | 涉及算法 |
|------|------|----------|
| `/` | 首页(智能穿搭推荐) | 朴素贝叶斯 + K-Means + 推荐穿搭图库 |
| `/clothing` | 服饰库(48 套实拍) | K-Means 聚类筛选 |
| `/my-clothes` | 个人衣服(上传管理) | — |
| `/recommendation` | 在线试衣 | 3D 形象 + 抠图合成 |
| `/trend-prediction` | **潮流预测** | **RandomForest 时序** |
| `/ai-assistant` | **AI 助手** | **DeepSeek/智谱/通义/OpenAI/Ollama** |
| `/weather` | 天气输入 → 推荐 | **朴素贝叶斯** |
| `/body-shape` | 身材信息 | 身材数据 CRUD |
| `/profile` | 我的 | 收藏/历史/偏好入口 |
| `/favorites` | 收藏(去重 + 取消) | — |
| `/history` | 历史浏览 | — |
| `/preferences` | 个人偏好设置 | — |
| `/login` / `/register` | 登录注册 | — |

### 新手引导
- 首次进入任意页面自动弹出(2 步:性别 + 风格,最多选 3 个)
- F5 刷新强制重弹;同会话内点链接不打扰
- 数据持久化到 localStorage + 游客 session

### 游客模式
- 无需注册,自动创建游客 user(幂等,cookie 保留 1 年)
- 收藏 / 历史 / 偏好 全部跨刷新保留

---

## 🔌 API 接口

### 推荐与算法
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/recommendations` | POST | 朴素贝叶斯 + K-Means 融合推荐(温度+天气+季节+用户) |
| `/api/recommendations/train` | POST | 重新训练所有模型 |
| `/api/recommendation-images` | GET | 获取 48 个推荐穿搭目录的所有图片 |
| `/api/trend-prediction` | POST | 潮流预测(timeRange: current/next-month/next-season) |
| `/api/generate-trend-images` | POST | 按预测风格从对应目录取图 |

### AI 助手
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/ai/providers` | GET | 返回 5 个 provider 状态(已配置/未配置) |
| `/api/ai-chat` | POST | 聊天接口(支持前端切换 provider/model) |

### 服饰与评价
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/clothing` | GET / POST | 服饰列表 / 新增 |
| `/api/clothing/<id>` | GET | 服饰详情 |
| `/api/clothing/<id>/comments` | GET | 获取细粒度评价(材质/版型/风格/搭配) |
| `/api/clothing/<id>/generate-comment` | POST | 现场生成一条评价 |
| `/api/clothing/generate-all-comments` | POST | 批量为所有服饰生成评价 |

### 用户与认证
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 注册 |
| `/api/auth/login` | POST | 登录 |
| `/api/auth/guest` | POST | 创建/获取游客(幂等) |
| `/api/auth/logout` | POST | 登出 |
| `/api/auth/status` | GET | 当前登录状态 |

### 收藏 / 历史 / 偏好
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/favorites` | POST | 收藏(支持穿搭方案 + 单品) |
| `/api/favorites/<user_id>` | GET | 列出收藏 |
| `/api/favorites/remove` | POST | 取消收藏(按签名) |
| `/api/browse-history` | POST | 记录浏览(穿搭/单品) |
| `/api/browse-history/<user_id>` | GET | 列出浏览历史 |
| `/api/browse-history/<history_id>` | DELETE | 删除一条 |
| `/api/browse-history/clear/<user_id>` | POST | 清空 |

### 天气 / 身材 / 通用
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/weather` | POST | 录入天气(触发朴素贝叶斯筛选) |
| `/api/weather/current` | GET | 最新天气 |
| `/api/body-shape` | GET / POST | 身材信息 |
| `/api/body-shape/calculate/<item_id>` | GET | 身材-服饰匹配度计算 |
| `/api/users` | POST | 创建用户 |
| `/api/users/<id>` | GET / PUT | 用户详情/更新 |
| `/health` | GET | 健康检查 |

---

## 🗄️ 数据库表

| 表名 | 作用 | 关键字段 |
|------|------|----------|
| `users` | 用户(含游客) | id, username, height, weight, skin_tone, style_preference, usual_scenes, is_guest |
| `weather` | 天气记录 | id, temperature, weather_condition, season, recorded_at |
| `clothing_items` | 服饰 | id, name, category, color, style, season, suitable_temperature_min/max, image_path, feature_vector, cluster_id |
| `user_favorites` | 收藏 | id, user_id, outfit_combo, weather_context, created_at |
| `recommendation_history` | 推荐历史 | id, user_id, weather_id, recommended_outfits, feedback |
| `body_shape` | 身材信息 | user_id, height, weight, 各部位围度/长度, skin_tone, body_type, gender |
| `browse_history` | 浏览历史 | id, user_id, item_type(outfit/clothing), item_id, title, image_url, meta, created_at |
| `clothing_comments` | 细粒度评价 | id, clothing_id, dimension(材质/版型/风格/搭配), content, generated_at |

---

## 🧮 推荐融合流程

`POST /api/recommendations` 内部流程:

```
用户输入 (温度, 天气, 季节, 用户ID)
       ↓
① 朴素贝叶斯 — 天气→服饰类别概率排序
       ↓
② K-Means — 对候选服饰按风格聚类(cluster_id 写回 DB)
       ↓
推荐结果 (top 5 穿搭方案) + 入库 recommendation_history
```

---

## 👥 分工

| 成员 | 负责模块 | 核心文件 |
|------|----------|----------|
| 成员1 | 朴素贝叶斯 | `algorithms/naive_bayes_classifier.py` + `data/weather_clothing_data.csv` |
| 成员2 | K-Means 聚类 | `algorithms/kmeans_clustering.py` + 图片特征 |
| 成员5 | 系统集成 + 全栈 | `app.py` / `routes/` / `services/` / `templates/` / `static/` / `database/` |
| 成员5 (新增) | **随机森林潮流预测** | `algorithms/trend_predictor.py` + `data/fashion_sales_history.csv` |

---

## 🎨 设计风格

- **色系**:米白 #FDFCFA · 暖灰背景 #F4F1ED · 深棕灰 #4A4540 · 灰蓝渐变 #9AA8B0→#7A8A92 · 暖红 #C97676
- **布局**:快看漫画风格 fixed 顶/底导航 + 大圆角卡片 + 渐变按钮
- **响应式**:适配手机 / 平板 / 桌面

---

## 🧪 测试

```bash
# 单元测试
python -m pytest tests/

# API 测试
python tests/test_api.py
```

---

## 📝 开发说明

### 添加新 ML 算法
1. 在 `algorithms/` 下创建 `<name>.py`,实现 `fit()` / `predict()` / `save()` / `load()` 接口
2. 在 `algorithms/__init__.py` 导出
3. 在 `services/recommendation_service.py` 接入融合流程
4. 在 `models/` 下保存 `.pkl`

### 添加新 LLM provider
1. 在 `services/llm_service.py` 的 `PROVIDERS` 字典加配置
2. 实现 `base_url` / `default_model` / `env_key`
3. 前端 `templates/ai_assistant.html` 的 `<select>` 加 `<option>`

### 添加新页面
1. 在 `templates/` 下创建 HTML,继承 `base.html`
2. 在 `app.py` 加 `@app.route('/path')`
3. 在 `base.html` 底部导航注册入口

---

## 📄 许可证

MIT License
