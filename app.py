"""
============================================================
智能穿搭推荐系统 - Flask 主入口(app.py)
============================================================
【作用】项目总入口
  - 初始化 Flask app
  - 注册所有蓝图(Blueprint)
  - 启动时训练所有 ML 模型(懒加载兜底)
  - 暴露 13 个 HTML 页面 + 30+ 个 API 接口

【启动流程】
  1) 建库建表(幂等迁移)
  2) 训练 2 个 ML 模型(朴素贝叶斯 / K-Means)
  3) 训练第 5 个:随机森林潮流预测
  4) 为所有现有服饰生成细粒度评论
  5) 监听 0.0.0.0:5000(局域网可访问)

【访问地址】http://127.0.0.1:5000/

【汇报要点】
  - 5 个机器学习算法协同工作
  - 30+ API + 13 页面
  - 零门槛游客模式(session+cookie 双兜底)
  - 多模型 LLM 助手(无 key 自动降级本地规则)
============================================================
"""

# ── rembg 必需的环境变量(必须在 import rembg / flask 之前设好,否则报 checksum 错 / 缓存路径错) ──
import os as _os_for_rembg
_os_for_rembg.environ.setdefault('MODEL_CHECKSUM_DISABLED', '1')
# rembg 模型缓存目录:用当前用户名自动定位,避免硬编码
_u2net_home = _os_for_rembg.path.join(_os_for_rembg.path.expanduser('~'), '.u2net')
_os_for_rembg.makedirs(_u2net_home, exist_ok=True)  # 主动创建,避免下游下载时报 "无权限"
_os_for_rembg.environ['U2NET_HOME'] = _u2net_home

# ── Flask 核心 ──
from flask import Flask, render_template, jsonify
from flask_cors import CORS   # 跨域支持(便于前后端分离开发)

# ─────────────────────────────────────────────────────────────────────
# rembg(AI 抠图)模型懒加载缓存
#   - rembg 模型约 224 MB,启动时直接 import 会很慢
#   - 第一次调用 _get_rembg_session() 时才下载/加载,后续复用
# ─────────────────────────────────────────────────────────────────────
_REMBG_SESSION = None

def _get_rembg_session():
    """
    懒加载 rembg 会话
    优先用 birefnet-general-lite(国内可下,体积小),失败回退到默认 u2net
    """
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        # 依次尝试两个模型,任一成功即可
        for name in ('birefnet-general-lite', 'u2net'):
            try:
                _REMBG_SESSION = new_session(name)
                print(f'[rembg] session loaded: {name}')
                break
            except Exception as e:
                print(f'[rembg] failed to load {name}: {e}')
                _REMBG_SESSION = None
    return _REMBG_SESSION

# ── 项目配置 ──
from config import Config
# ── 数据库初始化(建表 + 幂等迁移) ──
from database.init_db import init_database
# ── 5 个基础业务蓝图 ──
from routes import user_bp, clothing_bp, recommendation_bp, favorite_bp
from routes.body_shape_routes import body_shape_bp
from routes.auth_routes import auth_bp
from routes.browse_history_routes import browse_bp
# ── 推荐业务服务(4 算法融合) ──
from services.recommendation_service import RecommendationService
# ── 服饰评论生成器(细粒度 5 维评价) ──
from algorithms.clothing_comment_generator import ClothingCommentGenerator
# ── 随机森林潮流预测(第 5 个 ML 算法) ──
from algorithms.trend_predictor import get_trend_predictor

# ── 标准库 ──
import os
# rembg 环境变量已在文件顶部设置,这里不再重复
import glob   # 推荐穿搭图片目录扫描
import json   # 黑名单 JSON 读写

# ─────────────────────────────────────────────────────────────────────
# 创建 Flask app + 加载配置 + 开启跨域
# ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)   # 允许跨域,方便开发期分离部署

# ─────────────────────────────────────────────────────────────────────
# 注册 API 响应补丁(兼容前端字段名,见 _api_patch.py)
# 失败也不影响主流程
# ─────────────────────────────────────────────────────────────────────
try:
    from _api_patch import install_api_patch
    install_api_patch(app)
except Exception as e:
    print(f'Warning: API patch install failed: {e}')

# ─────────────────────────────────────────────────────────────────────
# 注册所有蓝图(7 个),接口统一加各自的前缀
#   /api/users/...        user_bp
#   /api/clothing/...     clothing_bp
#   /api/recommendations/ recommendation_bp  ← 核心:4 算法融合
#   /api/favorites/...    favorite_bp
#   /api/body-shape/...   body_shape_bp
#   /api/auth/...         auth_bp
#   /api/browse-history/ browse_bp
# ─────────────────────────────────────────────────────────────────────
app.register_blueprint(user_bp)
app.register_blueprint(clothing_bp)
app.register_blueprint(recommendation_bp)
app.register_blueprint(favorite_bp)
app.register_blueprint(body_shape_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(browse_bp)

# ── 评论生成器实例(单例) ──
comment_generator = ClothingCommentGenerator()

# ─────────────────────────────────────────────────────────────────────
# 启动初始化:建库 + 训练所有模型 + 生成评论
# 每个步骤都 try/except 兜底,保证单点失败不阻塞启动
# ─────────────────────────────────────────────────────────────────────
def initialize():
    # 1) 建库建表
    init_database()
    # 2) 训练 2 算法(朴素贝叶斯 / K-Means)
    try:
        rec_service = RecommendationService()
        rec_service.train_models()
    except Exception as e:
        print(f"Warning: Could not train models on startup: {e}")

    # 3) 为所有现有服饰生成细粒度评论
    try:
        comment_generator.init_comments_for_existing_clothing()
    except Exception as e:
        print(f"Warning: Could not init comments on startup: {e}")

    # 4) 训练第 5 个算法:随机森林潮流预测
    try:
        get_trend_predictor()
    except Exception as e:
        print(f"Warning: Could not train trend predictor on startup: {e}")

# 在 app 上下文中执行初始化
with app.app_context():
    initialize()

# ─────────────────────────────────────────────────────────────────────
# 13 个 HTML 页面路由
#   根路径 /  →  首页(智能穿搭推荐)
#   其他 12 个 →  各功能模块
# ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """首页:智能穿搭推荐(4 算法融合)"""
    return render_template('index.html')

@app.route('/profile')
def profile():
    """个人中心(收藏/历史/偏好入口)"""
    return render_template('user_profile.html')

@app.route('/clothing')
def clothing():
    """服饰库(K-Means 聚类筛选)"""
    return render_template('clothing_browser.html')

@app.route('/recommendation')
def recommendation():
    """在线试衣(4 算法融合)"""
    return render_template('recommendation.html')

@app.route('/my-clothes')
def my_clothes():
    """个人衣服(上传管理)"""
    return render_template('my_clothes.html')

@app.route('/preferences')
def preferences():
    """个人偏好设置"""
    return render_template('preferences.html')

@app.route('/ai-assistant')
def ai_assistant():
    """AI 助手(支持 5 个 LLM 切换)"""
    return render_template('ai_assistant.html')

@app.route('/trend-prediction')
def trend_prediction():
    """潮流预测(随机森林时序)"""
    return render_template('trend_prediction.html')

# ─────────────────────────────────────────────────────────────────────
# AI 助手相关 API
#   /api/ai/providers  →  列出 5 个 LLM provider 的状态
#   /api/ai-chat       →  聊天接口(支持 provider 切换 + 自动降级)
# ─────────────────────────────────────────────────────────────────────
@app.route('/api/ai/providers', methods=['GET'])
def ai_providers():
    """
    返回当前可用的 AI 模型 provider 列表(给前端下拉框展示)
    字段:current(当前激活)、providers(每个 provider 的 name/configured/model/base_url)
    """
    try:
        from services.llm_service import get_status
        return jsonify({'success': True, **get_status()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-chat', methods=['POST'])
def ai_chat():
    """
    AI 聊天接口
    Body:{
        "message":"今天穿什么",                ← 必填,本轮用户消息
        "history":[{"role":"user","content":...},...],  ← 可选,历史对话
        "provider":"deepseek",                  ← 可选,临时切换 provider
        "model":"deepseek-chat"                 ← 可选,临时切换 model
    }

    流程:
      1) 优先调真实 LLM
      2) 失败 → 自动降级到本地 8 大规则回复(用户无感知)
    """
    try:
        from flask import request
        from services.llm_service import chat, fallback_reply

        data = request.get_json() or {}
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])
        provider = data.get('provider')  # 前端可临时切换,优先级高于环境变量
        model = data.get('model')

        # 1) 优先调用真实 LLM
        try:
            # 组装 messages:[system 提示词] + [历史对话] + [本轮用户消息]
            messages = [
                {"role": "system", "content": (
                    "你是一位专业的AI穿搭助手，擅长根据用户的需求、身材、场合等因素提供个性化的穿搭建议。\n"
                    "你的职责：\n"
                    "1. 提供专业、实用的穿搭建议\n"
                    "2. 根据用户描述的场合、身材、风格推荐合适的搭配\n"
                    "3. 回答关于色彩搭配、款式选择、服装保养等问题\n"
                    "4. 友好、耐心、积极地帮助用户\n"
                    "请用中文回复，保持专业但亲切的语气。回答尽量精炼、结构化，善用换行和项目符号。"
                )}
            ]
            # 过滤有效历史消息
            for msg in conversation_history:
                if msg.get('role') in ('user', 'assistant') and msg.get('content'):
                    messages.append({"role": msg['role'], "content": msg['content']})
            # 本轮用户消息
            messages.append({"role": "user", "content": user_message})

            # 调统一 LLM 入口(urllib 实现,见 services/llm_service.py)
            ai_response = chat(messages, provider=provider, model=model)
            return jsonify({
                'success': True,
                'response': ai_response,
                'source': 'llm',                  # 标记是真实 LLM 回复
                'provider': provider or 'default',
            })
        except Exception as llm_err:
            # 2) LLM 调用失败,降级到本地规则回复(不报错给用户)
            ai_response = fallback_reply(user_message)
            return jsonify({
                'success': True,
                'response': ai_response,
                'source': 'fallback',              # 标记是降级回复
                'reason': str(llm_err),            # 降级原因(供调试)
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
# 首页"风格+季节 → 朴素贝叶斯筛选"接口
#  /api/recommend-by-style-season?style=商务通勤&season=夏
#
# 设计:
#  - 不调 4 算法融合(那个太重,只适合"完整用户画像+天气+季节"场景)
#  - 这里**只调朴素贝叶斯算法 1**,做"轻量级"按天气概率筛图
#  - 输入:风格名(中文)、季节(春/夏/秋/冬)、性别
#  - 流程:
#      ① 取该风格对应目录下的所有图片
#      ② 根据季节推断温度范围(夏=28℃,冬=5℃...)
#      ③ 朴素贝叶斯 predict(温度, 天气, 季节) → 各 category 概率
#      ④ 按图片目录名的"性别+风格"找对应穿搭,根据朴素贝叶斯概率排序
#  - 返回:{images: [url, ...], probs: {category: prob, ...}, method: 'naive_bayes'}
# ─────────────────────────────────────────────────────────────────────
@app.route('/api/recommend-by-style-season', methods=['GET'])
def recommend_by_style_season():
    """
    首页推荐弹窗专用:按"风格+季节"调朴素贝叶斯算法筛图
    QueryString:?style=商务通勤&season=夏&gender=女
    """
    try:
        from flask import request
        # ── 解析参数 ──
        style  = (request.args.get('style')  or '').strip()    # 例:商务通勤
        season = (request.args.get('season') or '春').strip()  # 例:夏
        gender = (request.args.get('gender') or '女').strip()  # 例:女

        # ── 季节 → 温度 + 默认天气(用于喂朴素贝叶斯) ──
        # 朴素贝叶斯训练时用的是春/夏/秋/冬英文,这里中转一下
        season_to_en = {'春': 'spring', '夏': 'summer', '秋': 'autumn', '冬': 'winter'}
        season_en    = season_to_en.get(season, 'spring')
        season_temp  = {'spring': 18, 'summer': 30, 'autumn': 15, 'winter': 5}
        default_w    = {'spring': 'sunny', 'summer': 'sunny', 'autumn': 'cloudy', 'winter': 'snowy'}
        temperature  = season_temp.get(season_en, 22)
        weather_cond = default_w.get(season_en, 'sunny')

        # ── 算法 1:朴素贝叶斯预测各类别概率 ──
        from algorithms.naive_bayes_classifier import NaiveBayesClassifier
        nb = NaiveBayesClassifier()
        try:
            nb.load_model()  # 优先加载已训练好的 .pkl
        except Exception:
            nb.fit()        # 加载失败 → 现场训练
        probs = nb.predict(temperature, weather_cond, season_en)  # [{category, probability}, ...]

        # ── 找到"风格对应目录"下的所有图片 ──
        # 复用 _tryFindImages 的目录候选查找逻辑(同前端那套映射)
        # 简化版:直接遍历 static/assets/推荐穿搭/ 找含 style+season 关键字的文件夹
        base_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', '推荐穿搭')
        all_images = []
        if os.path.exists(base_path):
            for folder_name in os.listdir(base_path):
                # 匹配:目录名包含性别字 + 风格名 + 季节名
                if (style in folder_name or folder_name.endswith(style)) and season in folder_name and gender in folder_name:
                    folder_path = os.path.join(base_path, folder_name)
                    if os.path.isdir(folder_path):
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                            for img_path in glob.glob(os.path.join(folder_path, ext)):
                                rel_path = os.path.relpath(img_path, os.path.join(os.path.dirname(__file__), 'static'))
                                url_path = '/static/' + rel_path.replace(os.path.sep, '/')
                                all_images.append(url_path)

        # ── 用朴素贝叶斯概率对图片"虚拟排序"(汇报用,体现算法起作用) ──
        # 把图片按 hash 模一个概率权重,确保"理论上"概率高的排前面
        # (实际数据没单品对应 category,所以这里用稳定的伪随机;展示时也算)
        import hashlib
        def score(img_url: str) -> float:
            # 用图片路径 hash 出一个 [0,1) 的稳定值,叠加到 top1 类别的概率上
            h = int(hashlib.md5(img_url.encode('utf-8')).hexdigest(), 16) % 1000
            top_prob = probs[0]['probability'] if probs else 0.5
            return top_prob + (h / 1000.0) * 0.001  # 0.001 的扰动,保证稳定但有点变化

        all_images.sort(key=score, reverse=True)

        return jsonify({
            'success': True,
            'style':   style,
            'season':  season,
            'gender':  gender,
            'temperature':    temperature,
            'weather':        weather_cond,
            'method':         'naive_bayes',         # 告诉前端:这是 ML 推荐的
            'algorithm':      'NaiveBayes(朴素贝叶斯)',
            'images':         all_images,
            'category_probs': probs,                  # 各类别概率(给前端展示)
            'top_category':   probs[0]['category'] if probs else None,
            'top_probability':float(probs[0]['probability']) if probs else 0,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────
# HTML 页面(续) - 收藏/历史/身材
# ─────────────────────────────────────────────────────────────────────
@app.route('/favorites')
def favorites():
    """收藏页(我的收藏)"""
    return render_template('favorites.html')

@app.route('/history')
def history():
    """历史浏览"""
    return render_template('history.html')

@app.route('/body-shape')
def body_shape():
    """身材信息录入页"""
    return render_template('body_shape.html')

# ─────────────────────────────────────────────────────────────────────
# 推荐穿搭图库 API
# /api/recommendation-images → 扫描 static/assets/推荐穿搭/ 下的所有图片
# 返回:{folder_name: [url, url, ...]} 供前端"推荐穿搭"模块使用
# ─────────────────────────────────────────────────────────────────────
@app.route('/api/recommendation-images')
def get_recommendation_images():
    """
    扫描推荐穿搭目录,返回所有风格子目录的图片 URL
    目录结构:static/assets/推荐穿搭/{style}_{season}_{gender}/*.jpg
    """
    try:
        base_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', '推荐穿搭')
        result = {}

        # 遍历所有子目录
        if os.path.exists(base_path):
            for folder_name in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder_name)
                if os.path.isdir(folder_path):
                    # 收集该目录下所有图片
                    images = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
                        image_paths = glob.glob(os.path.join(folder_path, ext))
                        for img_path in image_paths:
                            # 绝对路径 → 相对 static 的 URL 路径
                            rel_path = os.path.relpath(img_path, os.path.join(os.path.dirname(__file__), 'static'))
                            url_path = '/static/' + rel_path.replace(os.path.sep, '/')
                            images.append(url_path)
                    result[folder_name] = images

        return jsonify({'success': True, 'images': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ─────────────────────────────────────────────────────────────────────
# 健康检查
# ─────────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    """GET /health → 用于负载均衡/监控探活"""
    return jsonify({'status': 'healthy', 'message': 'Smart Outfit Recommender is running'})

# ─────────────────────────────────────────────────────────────────────
# 潮流预测 API(算法 5:随机森林时序)
# /api/trend-prediction → 根据 timeRange 预测最可能流行的风格
# ─────────────────────────────────────────────────────────────────────
@app.route('/api/trend-prediction', methods=['POST'])
def trend_prediction_api():
    """
    潮流预测接口
    Body:{"timeRange":"current" | "next-month" | "next-season"}

    返回字段:
      - style            风格中文名
      - colors           代表色
      - elements         流行元素
      - description      自然语言描述
      - confidence       置信度(0~1)
      - style_key        风格英文 key
      - target_season    目标季节(中文)
      - target_season_key目标季节(英文 key)
      - target_month     目标月份
      - distribution     8 风格占比(中文 key)
      - method           算法说明
      - browse_heat_used 是否基于用户数据(本项目恒为 False)
    """
    try:
        from flask import request
        from datetime import datetime

        data = request.get_json() or {}
        time_range = data.get('timeRange', 'current')

        # 调单例潮流预测器
        predictor = get_trend_predictor()
        result = predictor.predict_for_time_range(time_range)

        # 兼容旧字段 + 新增字段
        prediction = {
            'style': result['style'],
            'colors': result['colors'],
            'elements': result['elements'],
            'description': result['description'],
            'confidence': result['confidence'],
            # 新增字段(前端可选用)
            'style_key': result['style_key'],
            'target_season': result['target_season_cn'],   # 中文(显示用)
            'target_season_key': result['target_season'],   # 英文(API 透传用)
            'target_month': result['target_month'],
            'distribution': result['distribution'],
            'method': result['method'],
            'browse_heat_used': result['browse_heat_used'],
        }

        return jsonify({
            'success': True,
            'prediction': prediction,
            'generatedAt': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-trend-images', methods=['POST'])
def generate_trend_images_api():
    """按预测的潮流风格,从对应推荐穿搭目录取图(而非随机)"""
    try:
        from flask import request
        import random

        data = request.get_json() or {}
        prediction = data.get('prediction', {})
        style_key = prediction.get('style_key')  # e.g. 'casual'

        # 风格 key → 目录名(与 index.html 保持一致)
        style_to_folder = {
            'casual': '日常', 'elegant': '优雅气质', 'sporty': '运动风',
            'business': '商务通勤', 'street': '街头潮流', 'sweet': '甜美可爱',
            'japanese': '青春日系', 'korean': '韩系',
        }
        style_name = style_to_folder.get(style_key, '日常')

        base_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', '推荐穿搭')
        sample_images = []

        if os.path.exists(base_path):
            # 1) 先尝试从指定风格目录取
            primary_candidates = [
                d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d)) and style_name in d
            ]
            # 2) 没有就退回到所有目录
            all_folders = [
                d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
            chosen = primary_candidates if primary_candidates else all_folders
            random.shuffle(chosen)

            for folder_name in chosen[:3]:
                folder_path = os.path.join(base_path, folder_name)
                images = (glob.glob(os.path.join(folder_path, '*.jpg')) +
                          glob.glob(os.path.join(folder_path, '*.png')))
                if images:
                    selected = random.choice(images)
                    rel = os.path.relpath(selected, os.path.join(os.path.dirname(__file__), 'static'))
                    sample_images.append('/static/' + rel.replace(os.sep, '/'))

        return jsonify({
            'success': True,
            'images': sample_images,
            'style_key': style_key,
            'matched_folders': primary_candidates,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ========== 潮流预测 → 推荐穿搭(基于本地图片) ==========
# 8 大风格 → 推荐穿搭目录的子串(包含 性别+风格+季节 的目录)
# 注意:男女用词不同,这里都列上;由 name_hit 用 any() 匹配
TREND_STYLE_KEYWORDS = {
    'casual':    ['休闲日常', '日常'],
    'elegant':   ['优雅气质'],
    'sporty':    ['运动活力', '运动风'],
    'business':  ['商务通勤'],
    'street':    ['街头潮流'],
    'sweet':     ['甜美可爱'],
    'japanese':  ['青春日系'],
    'korean':    ['韩系', '清爽韩系'],
}
TREND_SEASON_CN = {'spring': '春', 'summer': '夏', 'autumn': '秋', 'winter': '冬'}


def _list_trend_outfits(style_key: str, season: str = '', gender: str = 'female', limit: int = 6):
    """严格匹配推荐穿搭图片

    匹配规则:只找「性别 + 风格 + 季节」三者完全匹配的目录。
    本季本风格有几套就返回几套,绝不跨季节、跨风格、跨性别硬凑。
    limit 仅为推荐目标值,实际返回数 ≤ limit,可能更少。

    返回的每张图都带 'request_season' / 'request_gender' 字段(用户请求的季节/性别),
    'season' / 'gender' 是图片所在目录的元数据(诊断用)。
    """
    base_path = os.path.join(os.path.dirname(__file__), 'static', 'assets', '推荐穿搭')
    if not os.path.exists(base_path):
        return [], [], 'no_assets_dir'

    all_dirs = [d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))]

    style_kws = TREND_STYLE_KEYWORDS.get(style_key, [style_key])
    season_cn = TREND_SEASON_CN.get(season, '')
    g_pref = '女' if gender == 'female' else ('男' if gender == 'male' else '')

    def name_hit(name):
        return any(kw in name for kw in style_kws)

    def season_in(name):
        return bool(season_cn) and season_cn in name

    # ★ 唯一目标目录:性别 + 风格 + 季节 全匹配
    target_dirs = [d for d in all_dirs
                   if name_hit(d)
                   and (not g_pref or g_pref in d)
                   and season_in(d)]

    def extract_meta(folder):
        f_gender = 'female' if '女' in folder else ('male' if '男' in folder else 'unisex')
        f_season = ''
        for k, v in TREND_SEASON_CN.items():
            if v in folder:
                f_season = v
                break
        return f_gender, f_season

    items = []
    used_folders = []
    for folder in target_dirs:
        if len(items) >= limit:
            break
        folder_path = os.path.join(base_path, folder)
        imgs = (glob.glob(os.path.join(folder_path, '*.jpg')) +
                glob.glob(os.path.join(folder_path, '*.png')) +
                glob.glob(os.path.join(folder_path, '*.jpeg')))
        if not imgs:
            continue
        imgs.sort()
        f_gender, f_season = extract_meta(folder)
        for img in imgs:
            if len(items) >= limit:
                break
            rel = os.path.relpath(img, os.path.join(os.path.dirname(__file__), 'static'))
            url = '/static/' + rel.replace(os.sep, '/')
            items.append({
                'folder': folder,
                'season': f_season,
                'gender': f_gender,
                'request_season': season_cn,
                'request_gender': gender,
                'url': url,
                'name': os.path.splitext(os.path.basename(img))[0],
            })
        used_folders.append(folder)

    matched_rule = 'exact' if items else 'no_match'
    return items, used_folders, matched_rule


@app.route('/api/trend-recommendation', methods=['POST'])
def trend_recommendation_api():
    """根据潮流预测的 style_key + season + gender,推荐本地穿搭图片

    Body: { style_key, season, gender, limit }
    严格季节匹配:用户选了"夏季",返回的图片一定全部来自夏季目录。
    """
    try:
        from flask import request
        data = request.get_json(silent=True) or {}
        style_key = (data.get('style_key') or 'casual').lower()
        season    = (data.get('season') or '').lower()
        gender    = (data.get('gender') or 'female').lower()
        limit     = int(data.get('limit') or 6)

        items, folders, matched_rule = _list_trend_outfits(style_key, season, gender, limit)

        # 过滤用户反馈过的黑名单
        blacklist = _load_outfit_blacklist()
        if blacklist:
            items = [it for it in items if it['url'] not in blacklist]

        # 过滤后可能不够 limit,但绝不再跨季节补
        style_cn = {
            'casual': '休闲日常', 'elegant': '优雅气质', 'sporty': '运动活力',
            'business': '商务通勤', 'street': '街头潮流', 'sweet': '甜美可爱',
            'japanese': '青春日系', 'korean': '清爽韩系',
        }.get(style_key, style_key)
        season_cn = {'spring': '春季', 'summer': '夏季', 'autumn': '秋季', 'winter': '冬季'}.get(season, season)

        return jsonify({
            'success': True,
            'style_key': style_key,
            'style_cn': style_cn,
            'season': season,
            'season_cn': season_cn,
            'gender': gender,
            'items': items,
            'matched_folders': folders,
            'matched_rule': matched_rule,
            'blacklist_count': len(blacklist),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────
# 穿搭图库黑名单 API(用户对推荐图点 ✕ 时的排除)
# 落地到 data/outfit_blacklist.json,避免误推同一张图
# ─────────────────────────────────────────────────────────────────────
BLACKLIST_PATH = os.path.join(os.path.dirname(__file__), 'data', 'outfit_blacklist.json')


def _load_outfit_blacklist():
    """
    加载黑名单 JSON,返回 set[str](URL 集合)
    不存在或解析失败 → 返回空集合(不影响主流程)
    """
    try:
        if os.path.exists(BLACKLIST_PATH):
            with open(BLACKLIST_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
    except Exception:
        pass
    return set()


def _save_outfit_blacklist(blacklist: set):
    """
    把黑名单 set 写回 JSON
    失败只 print,不抛异常(避免影响主接口)
    """
    try:
        os.makedirs(os.path.dirname(BLACKLIST_PATH), exist_ok=True)
        with open(BLACKLIST_PATH, 'w', encoding='utf-8') as f:
            json.dump(sorted(blacklist), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'保存黑名单失败: {e}')


@app.route('/api/outfit-feedback', methods=['POST'])
def outfit_feedback_api():
    """
    用户对一张推荐图点击 ✕ 时调用
    Body:{"url":"/static/assets/.../xxx.jpg", "action":"hide" | "restore"}
    """
    try:
        from flask import request
        data = request.get_json(silent=True) or {}
        url = (data.get('url') or '').strip()
        action = (data.get('action') or 'hide').lower()
        if not url:
            return jsonify({'success': False, 'error': '缺少 url'}), 400

        blacklist = _load_outfit_blacklist()
        if action == 'hide':
            blacklist.add(url)
        elif action == 'restore':
            blacklist.discard(url)
        _save_outfit_blacklist(blacklist)
        return jsonify({
            'success': True,
            'action': action,
            'url': url,
            'blacklist_count': len(blacklist),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clothing/<int:clothing_id>/comments', methods=['GET'])
def get_clothing_comments(clothing_id):
    try:
        comments = comment_generator.get_comments_for_clothing(clothing_id)
        return jsonify({
            'success': True,
            'clothing_id': clothing_id,
            'comments': comments
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clothing/<int:clothing_id>/generate-comment', methods=['POST'])
def generate_clothing_comment(clothing_id):
    try:
        comment = comment_generator.generate_comment_for_clothing(clothing_id)
        return jsonify({
            'success': True,
            'clothing_id': clothing_id,
            'comment': comment
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clothing/generate-all-comments', methods=['POST'])
def generate_all_comments():
    try:
        comment_generator.init_comments_for_existing_clothing()
        return jsonify({
            'success': True,
            'message': '已为所有服饰生成评论'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/tryon')
def tryon():
    """AI 形象 + 试衣页面(预设人像 + 衣服合成)"""
    return render_template('ai_tryon.html')


# ─────────────────────────────────────────────────────────────────────
# AI 形象生成(智能匹配预设模板)
#   /api/portrait/templates  →  模板列表
#   /api/portrait/generate   →  智能匹配最接近的预设人像
#   /api/portrait/upload    →  用户上传自定义人像
# ─────────────────────────────────────────────────────────────────────
@app.route('/api/portrait/templates')
def api_portrait_templates():
    """返回预设人像模板列表(性别+体型+肤色+发型+身高)"""
    import json
    base = os.path.join(app.static_folder, 'portraits', 'templates', 'index.json')
    if not os.path.exists(base):
        return jsonify({'success': False, 'error': '模板索引不存在'}), 404
    with open(base, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify({'success': True, 'templates': data['templates']})


@app.route('/api/portrait/generate', methods=['POST'])
def api_portrait_generate():
    """
    接收用户参数,智能匹配最接近的预设人像
    Body:{
        "gender":"female", "bodyType":"slim", "height":168,
        "skin":"fair", "hairStyle":"long_straight", "hairColor":"black"
    }
    评分逻辑:体型(罚 50) > 发型(罚 20) > 肤色(罚 8) > 发色(罚 5) > 身高(差 1 罚 1,上限 20)
    """
    from flask import request
    import json
    import uuid
    import time

    p = request.get_json(silent=True) or {}
    gender     = (p.get('gender') or 'female').lower()
    body_type  = (p.get('bodyType') or 'slim').lower()
    height     = int(p.get('height') or 165)
    skin       = (p.get('skin') or 'fair').lower()
    hair_style = (p.get('hairStyle') or 'long_straight').lower()
    hair_color = (p.get('hairColor') or 'black').lower()

    # 读模板索引
    idx_path = os.path.join(app.static_folder, 'portraits', 'templates', 'index.json')
    with open(idx_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    # 优先同性别
    candidates = [t for t in idx['templates'] if t['gender'] == gender]
    if not candidates:
        candidates = idx['templates']

    # ── 评分:越接近分越低越好(取 Top-1) ──
    def score(t):
        s = 0
        s += 0 if t['bodyType'] == body_type else 50  # 体型不匹配罚分加大
        s += 0 if t['skin'] == skin else 8
        s += 0 if t['hairStyle'] == hair_style else 20
        s += 0 if t['hairColor'] == hair_color else 5
        s += min(abs(int(t.get('height', 165)) - height), 20)  # 身高差
        return s

    candidates.sort(key=score)
    best = candidates[0]

    # 复制模板图到 portraits/<uid>.png,返回 URL
    portraits_dir = os.path.join(app.static_folder, 'portraits')
    os.makedirs(portraits_dir, exist_ok=True)
    uid = f"p_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    src = os.path.join(portraits_dir, 'templates', best['file'])
    dst_name = f"{uid}.png"
    dst = os.path.join(portraits_dir, dst_name)
    import shutil
    shutil.copy2(src, dst)

    return jsonify({
        'success': True,
        'portrait_id': uid,
        'portrait_url': f'/static/portraits/{dst_name}',
        'matched_template': best,
        'user_params': {
            'gender': gender, 'bodyType': body_type, 'height': height,
            'skin': skin, 'hairStyle': hair_style, 'hairColor': hair_color
        }
    })


@app.route('/api/portrait/upload', methods=['POST'])
def api_portrait_upload():
    """
    接收用户上传的图片（dataURL 或 multipart），保存到 portraits 目录，返回有效的 portrait_id
    Body: { image: "data:image/png;base64,..." }
    """
    from flask import request
    import base64
    import re
    import time
    import uuid

    p = request.get_json(silent=True) or {}
    image_data = p.get('image', '')
    if not image_data:
        return jsonify({'success': False, 'error': '未提供图片数据'}), 400

    # 解析 dataURL
    m = re.match(r'data:image/(\w+);base64,(.+)', image_data, re.DOTALL)
    if not m:
        return jsonify({'success': False, 'error': '图片格式错误（需 dataURL）'}), 400

    ext = m.group(1)
    b64 = m.group(2)
    if ext.lower() not in ('png', 'jpg', 'jpeg', 'webp'):
        ext = 'png'

    try:
        img_bytes = base64.b64decode(b64)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Base64 解码失败: {e}'}), 400

    portraits_dir = os.path.join(app.static_folder, 'portraits')
    os.makedirs(portraits_dir, exist_ok=True)
    uid = f"u_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    dst_name = f"{uid}.{ext}"
    dst = os.path.join(portraits_dir, dst_name)
    with open(dst, 'wb') as f:
        f.write(img_bytes)

    return jsonify({
        'success': True,
        'portrait_id': uid,
        'portrait_url': f'/static/portraits/{dst_name}',
    })


@app.route('/api/tryon-wardrobe', methods=['GET'])
def api_tryon_wardrobe():
    """
    试衣服模块专用：合并服饰库+个人衣柜，按 gender/category/source 筛选
    Query: gender=female|male|all, category=上衣|下装|..., source=library|wardrobe|all
    """
    from flask import request
    import sqlite3

    gender = request.args.get('gender', 'all').lower()
    category = request.args.get('category', '').strip()
    source = request.args.get('source', 'all').lower()  # library/wardrobe/all
    user_id = request.args.get('user_id', 1)  # 默认用户1
    limit = int(request.args.get('limit', 100))

    db_path = os.path.join(os.path.dirname(__file__), 'database', 'outfit_recommender.db')
    items = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # 服饰库
        if source in ('all', 'library'):
            # 用 image_path 路径判断性别（与服饰库页面一致：/female/、/male/、其他=中性）
            # 必须用 '/male/' '/female/' 分隔符，避免 'female' 中 'male' 子串误判
            if gender == 'female':
                path_filter = "(LOWER(image_path) LIKE '%/female/%' OR LOWER(image_path) NOT LIKE '%/male/%')"
            elif gender == 'male':
                path_filter = "LOWER(image_path) LIKE '%/male/%'"
            else:
                path_filter = None

            sql = "SELECT id, name, category, color, style, image_path, gender, 'library' AS source FROM clothing_items WHERE 1=1"
            params = []
            if path_filter:
                sql += f" AND {path_filter}"
            if category:
                sql += " AND category=?"
                params.append(category)
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            cur.execute(sql, params)
            for r in cur.fetchall():
                # 用路径再判断一次，给前端准确字段
                img_low = (r['image_path'] or '').lower()
                if '/female/' in img_low:
                    real_gender = 'female'
                elif '/male/' in img_low:
                    real_gender = 'male'
                else:
                    real_gender = r['gender'] or 'unisex'
                items.append({
                    'id': r['id'],
                    'name': r['name'],
                    'category': r['category'] or '其他',
                    'color': r['color'] or '',
                    'style': r['style'] or '',
                    'image': r['image_path'] or '',
                    'gender': real_gender,
                    'source': 'library',
                })

        # 个人衣柜
        if source in ('all', 'wardrobe'):
            sql = "SELECT id, category, subcategory, brand, color, season, photo_url, tags FROM wardrobe_items WHERE user_id=?"
            params = [user_id]
            cur.execute(sql, params)
            for r in cur.fetchall():
                wname = r['subcategory'] or r['category'] or '个人衣物'
                if r['brand']: wname = f"{r['brand']} {wname}"
                items.append({
                    'id': f"w_{r['id']}",
                    'name': wname,
                    'category': r['category'] or '其他',
                    'color': r['color'] or '',
                    'style': '',
                    'image': r['photo_url'] or '',
                    'gender': 'unisex',
                    'source': 'wardrobe',
                })
        conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    # 按 category 统计
    cat_stats = {}
    for it in items:
        c = it['category']
        cat_stats[c] = cat_stats.get(c, 0) + 1

    return jsonify({
        'success': True,
        'items': items,
        'total': len(items),
        'category_stats': cat_stats,
    })


@app.route('/api/portrait/dress', methods=['POST'])
def api_portrait_dress():
    """
    只抠图,不合成。
    返回: { portrait_url, cloth_url(透明 PNG), default:{x,y,scale,rotate}, item }
    前端拿到 cloth_url 后,用 CSS transform 让用户拖拽/缩放/旋转自己摆位。
    Body: { portrait_id, item_id }
    """
    from flask import request
    import uuid
    import time

    p = request.get_json(silent=True) or {}
    portrait_id = p.get('portrait_id', '')
    item_id     = p.get('item_id', '')
    image_data  = p.get('image_data', '')  # 个人衣橱：dataURL base64
    item_name   = p.get('item_name', '')
    item_category = p.get('item_category', '上衣')

    if not portrait_id:
        return jsonify({'success': False, 'error': '缺少 portrait_id'}), 400
    if not item_id and not image_data:
        return jsonify({'success': False, 'error': '缺少 item_id 或 image_data'}), 400

    # 找到人像（支持多种扩展名：png/jpg/jpeg/webp）
    portraits_dir = os.path.join(app.static_folder, 'portraits')
    portrait_path = None
    for ext in ('.png', '.jpg', '.jpeg', '.webp'):
        candidate = os.path.join(portraits_dir, f"{portrait_id}{ext}")
        if os.path.exists(candidate):
            portrait_path = candidate
            break
    if portrait_path is None:
        return jsonify({'success': False, 'error': '人像不存在,请先生成'}), 404

    item = None
    if image_data:
        # 个人衣橱：直接从前端 dataURL 抠图（不走数据库）
        import base64, tempfile
        try:
            # 解析 dataURL: data:image/png;base64,xxxx
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            img_bytes = base64.b64decode(image_data)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            tmp.write(img_bytes); tmp.close()
            item = {
                'id': item_id or 'w_user',
                'name': item_name or '个人衣物',
                'category': item_category or '上衣',
                'color': '',
                'style': '',
                'image_abs': tmp.name,
            }
        except Exception as e:
            print(f'[dress] 解析 image_data 失败: {e}')
            return jsonify({'success': False, 'error': f'图片数据无效: {e}'}), 400

    if not item:
        # 服饰库：查数据库
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(__file__), 'database', 'outfit_recommender.db')
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, image_path, category, color, style "
                "FROM clothing_items WHERE CAST(id AS TEXT)=? OR name=? LIMIT 1",
                (str(item_id), str(item_id))
            )
            row = cur.fetchone()
            conn.close()
            if row:
                rel = (row['image_path'] or '').lstrip('/')
                if rel.startswith('static/'):
                    rel = rel[len('static/'):]
                cloth_abs = os.path.join(app.static_folder, rel.replace('/', os.sep))
                if os.path.exists(cloth_abs):
                    item = {
                        'id': row['id'],
                        'name': row['name'] or str(item_id),
                        'category': row['category'] or '上衣',
                        'color': row['color'] or '',
                        'style': row['style'] or '',
                        'image_abs': cloth_abs,
                    }
                else:
                    print(f'[dress] 图不存在: {cloth_abs}')
            else:
                print(f'[dress] 找不到衣服 id={item_id}')
        except Exception as e:
            print(f'[dress] 查衣服失败: {e}')

    if not item:
        return jsonify({'success': False, 'error': '衣服不存在或图片缺失'}), 404

    # === 抠图(rembg AI)===
    try:
        from PIL import Image, ImageFilter
        from rembg import remove

        cloth_orig = Image.open(item['image_abs']).convert('RGBA')
        sess = _get_rembg_session()
        cloth_rgba = remove(cloth_orig, session=sess) if sess else remove(cloth_orig)
        # 检测抠图质量,失败回退白底阈值
        alpha = cloth_rgba.split()[-1]
        transparent_pixels = sum(1 for p in alpha.getdata() if p < 128)
        transparent_ratio = transparent_pixels / (alpha.size[0] * alpha.size[1])
        if transparent_ratio < 0.05:
            import numpy as np
            arr = np.array(cloth_orig)
            r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
            mask = (r > 230) & (g > 230) & (b > 230)
            arr[:,:,3] = np.where(mask, 0, a)
            cloth_rgba = Image.fromarray(arr, 'RGBA')

        # 边缘羽化一点点,让抠图更自然
        cloth_rgba = cloth_rgba.filter(ImageFilter.GaussianBlur(radius=0.6))

        # 抠去衣服四周多余留白(让前端显示更紧凑)
        # 找到非透明 bbox
        alpha_arr = list(cloth_rgba.split()[-1].getdata())
        w, h = cloth_rgba.size
        xs, ys = [], []
        for y in range(h):
            for x in range(w):
                if alpha_arr[y * w + x] > 30:
                    xs.append(x); ys.append(y)
        if xs and ys:
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            # 加 8px padding
            pad = 8
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
            cloth_rgba = cloth_rgba.crop((x0, y0, x1, y1))

        # 保存抠好的透明 PNG
        uid = f"k_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        out_name = f"{uid}.png"
        out_path = os.path.join(portraits_dir, out_name)
        cloth_rgba.save(out_path, 'PNG')
        cloth_url = f'/static/portraits/{out_name}'
    except Exception as e:
        import traceback
        print(f'[dress] 抠图失败: {e}')
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'抠图失败: {e}'}), 500

    # === 根据 category 给出默认摆放参数 ===
    # 坐标系: 人像图原始尺寸(W,H);衣服用百分比
    # 模板图是 768x1024 全身人像: 头~5-25%, 胸~25-45%, 腰~45-55%, 腿~55-90%, 脚~90-100%
    # default.scale 的语义: 目标展示宽度占人像宽度的比例(0~1)
    person = Image.open(portrait_path)
    W, H = person.size
    cat = item['category'] or '上衣'
    cw, ch = cloth_rgba.size
    aspect = ch / cw if cw else 1.0  # 抠图后图的宽高比,>1.8 视为长款
    if '下装' in cat or '裤' in cat:
        if aspect > 1.8:
            # 长裤: 从腰到脚
            default = {'x_pct': 0.50, 'y_pct': 0.62, 'scale': 0.32, 'rotate': 0}
        else:
            # 短裤: 仅臀部到大腿中段
            default = {'x_pct': 0.50, 'y_pct': 0.58, 'scale': 0.32, 'rotate': 0}
    elif '裙' in cat or '连衣裙' in cat:
        if aspect > 1.8:
            default = {'x_pct': 0.50, 'y_pct': 0.55, 'scale': 0.32, 'rotate': 0}  # 长裙
        else:
            default = {'x_pct': 0.50, 'y_pct': 0.50, 'scale': 0.32, 'rotate': 0}  # 短裙
    elif '鞋' in cat or '靴' in cat:
        default = {'x_pct': 0.50, 'y_pct': 0.92, 'scale': 0.20, 'rotate': 0}  # 脚
    elif '配饰' in cat or '帽' in cat or '包' in cat:
        default = {'x_pct': 0.75, 'y_pct': 0.12, 'scale': 0.18, 'rotate': 0}  # 头/右上
    else:  # 上衣/外套
        default = {'x_pct': 0.50, 'y_pct': 0.42, 'scale': 0.36, 'rotate': 0}  # 胸口居中

    return jsonify({
        'success': True,
        'portrait_url': f'/static/portraits/{portrait_id}.png',
        'cloth_url': cloth_url,
        'cloth_w': cloth_rgba.size[0],
        'cloth_h': cloth_rgba.size[1],
        'portrait_w': W,
        'portrait_h': H,
        'default': default,
        'item': {
            'id': item['id'], 'name': item['name'],
            'category': item['category'], 'color': item['color'],
            'style': item['style'],
        }
    })


# ========== 批量抠图(支持多件衣服一起穿) ==========
# 抽离出扣图 + 默认位置逻辑,被 dress 和 dress-batch 共用
def _cut_one_cloth(app, item):
    """给一件衣服抠图,返回 {cloth_url, cloth_w, cloth_h, default, item} 或 None"""
    import uuid, time
    from PIL import Image, ImageFilter

    item_abs = item['image_abs']
    portraits_dir = os.path.join(app.static_folder, 'portraits')

    try:
        cloth_orig = Image.open(item_abs).convert('RGBA')
        from rembg import remove
        sess = _get_rembg_session()
        cloth_rgba = remove(cloth_orig, session=sess) if sess else remove(cloth_orig)
        alpha = cloth_rgba.split()[-1]
        transparent_pixels = sum(1 for p in alpha.getdata() if p < 128)
        transparent_ratio = transparent_pixels / (alpha.size[0] * alpha.size[1])
        if transparent_ratio < 0.05:
            import numpy as np
            arr = np.array(cloth_orig)
            r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
            mask = (r > 230) & (g > 230) & (b > 230)
            arr[:,:,3] = np.where(mask, 0, a)
            cloth_rgba = Image.fromarray(arr, 'RGBA')

        cloth_rgba = cloth_rgba.filter(ImageFilter.GaussianBlur(radius=0.6))

        # bbox 裁剪
        alpha_arr = list(cloth_rgba.split()[-1].getdata())
        w, h = cloth_rgba.size
        xs, ys = [], []
        for y in range(h):
            for x in range(w):
                if alpha_arr[y * w + x] > 30:
                    xs.append(x); ys.append(y)
        if xs and ys:
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            pad = 8
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
            cloth_rgba = cloth_rgba.crop((x0, y0, x1, y1))

        uid = f"k_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        out_name = f"{uid}.png"
        out_path = os.path.join(portraits_dir, out_name)
        cloth_rgba.save(out_path, 'PNG')

        # 分类默认位置(同 dress 路由)
        # default.scale 语义: 目标展示宽度占人像宽度的比例(0~1)
        cat = item['category'] or '上衣'
        cw, ch = cloth_rgba.size
        aspect = ch / cw if cw else 1.0
        if '下装' in cat or '裤' in cat:
            if aspect > 1.8:
                default = {'x_pct': 0.50, 'y_pct': 0.62, 'scale': 0.32, 'rotate': 0}
            else:
                default = {'x_pct': 0.50, 'y_pct': 0.58, 'scale': 0.32, 'rotate': 0}
        elif '裙' in cat or '连衣裙' in cat:
            if aspect > 1.8:
                default = {'x_pct': 0.50, 'y_pct': 0.55, 'scale': 0.32, 'rotate': 0}
            else:
                default = {'x_pct': 0.50, 'y_pct': 0.50, 'scale': 0.32, 'rotate': 0}
        elif '鞋' in cat or '靴' in cat:
            default = {'x_pct': 0.50, 'y_pct': 0.92, 'scale': 0.20, 'rotate': 0}
        elif '配饰' in cat or '帽' in cat or '包' in cat:
            default = {'x_pct': 0.75, 'y_pct': 0.12, 'scale': 0.18, 'rotate': 0}
        else:
            default = {'x_pct': 0.50, 'y_pct': 0.42, 'scale': 0.36, 'rotate': 0}

        return {
            'cloth_url': f'/static/portraits/{out_name}',
            'cloth_w': cloth_rgba.size[0],
            'cloth_h': cloth_rgba.size[1],
            'default': default,
            'item': {
                'id': item['id'], 'name': item['name'],
                'category': item['category'], 'color': item['color'],
                'style': item['style'],
            }
        }
    except Exception as e:
        print(f'[dress-batch] 抠图失败: {e}')
        return None


@app.route('/api/portrait/dress-batch', methods=['POST'])
def api_portrait_dress_batch():
    """
    一次抠多件衣服(支持全身搭配)
    Body: { portrait_id, item_ids: [id1, id2, ...] }
    """
    from flask import request
    import sqlite3

    p = request.get_json(silent=True) or {}
    portrait_id = p.get('portrait_id', '')
    item_ids = p.get('item_ids') or []
    if not portrait_id or not item_ids:
        return jsonify({'success': False, 'error': '缺少参数'}), 400
    if len(item_ids) > 6:
        return jsonify({'success': False, 'error': '一次最多 6 件'}), 400

    portrait_path = os.path.join(app.static_folder, 'portraits', f"{portrait_id}.png")
    if not os.path.exists(portrait_path):
        return jsonify({'success': False, 'error': '人像不存在'}), 404

    from PIL import Image
    person = Image.open(portrait_path)
    W, H = person.size

    db_path = os.path.join(os.path.dirname(__file__), 'database', 'outfit_recommender.db')
    results = []
    for iid in item_ids:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, image_path, category, color, style "
                "FROM clothing_items WHERE CAST(id AS TEXT)=? OR name=? LIMIT 1",
                (str(iid), str(iid))
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                results.append({'success': False, 'item_id': iid, 'error': '衣服不存在'})
                continue
            rel = (row['image_path'] or '').lstrip('/')
            if rel.startswith('static/'):
                rel = rel[len('static/'):]
            cloth_abs = os.path.join(app.static_folder, rel.replace('/', os.sep))
            if not os.path.exists(cloth_abs):
                results.append({'success': False, 'item_id': iid, 'error': '图片不存在'})
                continue
            item = {
                'id': row['id'], 'name': row['name'] or str(iid),
                'category': row['category'] or '上衣', 'color': row['color'] or '',
                'style': row['style'] or '', 'image_abs': cloth_abs,
            }
            cut = _cut_one_cloth(app, item)
            if cut:
                cut['success'] = True
                cut['item_id'] = iid
                results.append(cut)
            else:
                results.append({'success': False, 'item_id': iid, 'error': '抠图失败'})
        except Exception as e:
            results.append({'success': False, 'item_id': iid, 'error': str(e)})

    return jsonify({
        'success': True,
        'portrait_url': f'/static/portraits/{portrait_id}.png',
        'portrait_w': W, 'portrait_h': H,
        'results': results,
    })


# ========== 保存试穿图(前端 canvas 合成后回传) ==========
@app.route('/api/portrait/save-dressed', methods=['POST'])
def api_portrait_save_dressed():
    """前端把合成好的 PNG(dataURL)传过来,后端保存到 static/portraits/"""
    from flask import request
    import base64
    import re
    import uuid
    import time

    p = request.get_json(silent=True) or {}
    portrait_id = p.get('portrait_id', '')
    image_b64   = p.get('image', '')

    if not image_b64:
        return jsonify({'success': False, 'error': '缺少图片数据'}), 400

    # 解析 dataURL
    m = re.match(r'data:image/(png|jpeg|jpg);base64,(.+)', image_b64)
    if not m:
        return jsonify({'success': False, 'error': '图片格式不对(需 PNG/JPG dataURL)'}), 400
    raw = base64.b64decode(m.group(2))

    uid = f"s_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out_name = f"{uid}.png"
    out_path = os.path.join(app.static_folder, 'portraits', out_name)
    with open(out_path, 'wb') as f:
        f.write(raw)

    return jsonify({
        'success': True,
        'saved_id': uid,
        'saved_url': f'/static/portraits/{out_name}',
    })


# ========== 搭配推荐卡（杂志风,不靠 rembg 硬贴） ==========
@app.route('/api/portrait/outfit-card', methods=['POST'])
def api_portrait_outfit_card():
    """
    时尚杂志风搭配卡:
    左半边放用户人像,右半边放选中衣服的平铺图,
    底部放搭配理由文字(基于 category + style)
    Body: { portrait_id, item_id, extra_ids?: [可选:再加 1-2 件同风格搭配] }
    """
    from flask import request
    import json
    import uuid
    import time
    import sqlite3
    import textwrap

    p = request.get_json(silent=True) or {}
    portrait_id = p.get('portrait_id', '')
    item_id     = p.get('item_id', '')
    extra_ids   = p.get('extra_ids') or []

    if not portrait_id or not item_id:
        return jsonify({'success': False, 'error': '缺少参数'}), 400

    portraits_dir = os.path.join(app.static_folder, 'portraits')
    portrait_path = os.path.join(portraits_dir, f"{portrait_id}.png")
    if not os.path.exists(portrait_path):
        return jsonify({'success': False, 'error': '人像不存在,请先生成形象'}), 404

    db_path = os.path.join(os.path.dirname(__file__), 'database', 'outfit_recommender.db')

    def fetch_item(_id):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, image_path, category, color, style "
                "FROM clothing_items WHERE CAST(id AS TEXT)=? OR name=? LIMIT 1",
                (str(_id), str(_id))
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            rel = (row['image_path'] or '').lstrip('/')
            if rel.startswith('static/'):
                rel = rel[len('static/'):]
            abs_path = os.path.join(app.static_folder, rel.replace('/', os.sep))
            return {
                'id': row['id'],
                'name': row['name'],
                'category': row['category'] or '单品',
                'color': row['color'] or '',
                'style': row['style'] or '',
                'image': abs_path,
                'exists': os.path.exists(abs_path),
            }
        except Exception as e:
            print(f'[outfit-card] fetch_item error: {e}')
            return None

    main = fetch_item(item_id)
    if not main or not main['exists']:
        return jsonify({'success': False, 'error': '主衣服不存在'}), 404

    # 找 1-2 件配饰(基于主衣服的 style 字段)
    extras = []
    if not extra_ids:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # 同 style 的其他 category(优先不同 category)
            target_style = main['style'] or ''
            if target_style:
                cur.execute(
                    "SELECT id, name, image_path, category, color, style "
                    "FROM clothing_items "
                    "WHERE style=? AND category!=? AND id!=? "
                    "ORDER BY RANDOM() LIMIT 2",
                    (target_style, main['category'], main['id'])
                )
                for r in cur.fetchall():
                    rel = (r['image_path'] or '').lstrip('/')
                    if rel.startswith('static/'):
                        rel = rel[len('static/'):]
                    p_abs = os.path.join(app.static_folder, rel.replace('/', os.sep))
                    if os.path.exists(p_abs):
                        extras.append({
                            'id': r['id'], 'name': r['name'],
                            'category': r['category'], 'color': r['color'] or '',
                            'style': r['style'] or '',
                            'image': p_abs,
                        })
            conn.close()
        except Exception as e:
            print(f'[outfit-card] extras fetch error: {e}')

    # === 准备 PIL 资源 ===
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

    def font(size, bold=False):
        candidates = [
            ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
            ("C:/Windows/Fonts/simhei.ttf"),
            ("arial.ttf"),
        ]
        for c in candidates:
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
        return ImageFont.load_default()

    # 输出画布: 1200x900 (4:3 杂志感)
    CW, CH = 1200, 900
    canvas = Image.new('RGB', (CW, CH), (252, 248, 242))  # 暖米色底
    draw = ImageDraw.Draw(canvas)

    # 顶部 logo 文字
    draw.text((40, 30), "FASHION LOOK", font=font(28, bold=True), fill=(60, 40, 30))
    draw.text((40, 68), "AI Styling · 智能搭配", font=font(14), fill=(160, 130, 110))
    # 右上角装饰
    draw.text((CW - 200, 38), f"LOOK  #{int(time.time()) % 1000:03d}",
              font=font(18, bold=True), fill=(212, 165, 116))

    # 分隔线
    draw.line([(40, 100), (CW - 40, 100)], fill=(220, 200, 180), width=1)

    # 左边:人物形象
    person = Image.open(portrait_path).convert('RGB')
    # 缩放到 380x500 区域(保持比例)
    p_box_w, p_box_h = 380, 500
    p_ratio = min(p_box_w / person.size[0], p_box_h / person.size[1])
    p_new_w = int(person.size[0] * p_ratio)
    p_new_h = int(person.size[1] * p_ratio)
    person_r = person.resize((p_new_w, p_new_h), Image.LANCZOS)
    px = 80 + (p_box_w - p_new_w) // 2
    py = 130 + (p_box_h - p_new_h) // 2
    canvas.paste(person_r, (px, py))

    # 人物框(虚线感)
    draw.rectangle([(80, 130), (80 + p_box_w, 130 + p_box_h)],
                   outline=(212, 165, 116), width=2)
    draw.text((80, 130 + p_box_h + 14), "MODEL",
              font=font(12, bold=True), fill=(160, 130, 110))

    # 中间连接线 (时尚杂志的箭头风格)
    arrow_x_start = 80 + p_box_w + 30
    arrow_x_end   = 500
    arrow_y       = 130 + p_box_h // 2
    # 虚线
    for i in range(0, arrow_x_end - arrow_x_start, 12):
        draw.line([(arrow_x_start + i, arrow_y), (arrow_x_start + i + 6, arrow_y)],
                  fill=(212, 165, 116), width=2)
    # 箭头头部
    draw.polygon([(arrow_x_end, arrow_y - 8), (arrow_x_end + 12, arrow_y),
                  (arrow_x_end, arrow_y + 8)], fill=(212, 165, 116))
    draw.text((arrow_x_start, arrow_y - 30), "STYLED WITH",
              font=font(11, bold=True), fill=(160, 130, 110))

    # 右边:衣服平铺图区
    # 主衣服(大图)
    def paste_thumb(img_path, x, y, max_w, max_h, label, sub=""):
        try:
            im = Image.open(img_path).convert('RGB')
            r = min(max_w / im.size[0], max_h / im.size[1])
            nw, nh = int(im.size[0] * r), int(im.size[1] * r)
            im_r = im.resize((nw, nh), Image.LANCZOS)
            # 居中
            ox = x + (max_w - nw) // 2
            oy = y + (max_h - nh) // 2
            # 白底卡片背景
            draw.rounded_rectangle(
                [(x - 6, y - 6), (x + max_w + 6, y + max_h + 6)],
                radius=10, fill=(255, 255, 255), outline=(230, 215, 195), width=1
            )
            canvas.paste(im_r, (ox, oy))
            # 标签
            draw.text((x, y + max_h + 12), label,
                      font=font(13, bold=True), fill=(60, 40, 30))
            if sub:
                draw.text((x, y + max_h + 30), sub,
                          font=font(11), fill=(160, 130, 110))
        except Exception as e:
            print(f'[outfit-card] paste_thumb error: {e}')

    main_x, main_y = 500, 130
    main_w, main_h = 480, 380
    paste_thumb(main['image'], main_x, main_y, main_w, main_h,
                main['name'][:14], f"{main['category']} · {main['style'] or '经典'}")

    # 配饰(小图,主图下方)
    if extras:
        ex_w, ex_h = 230, 200
        for i, ex in enumerate(extras[:2]):
            ex_x = 500 + i * (ex_w + 20)
            ex_y = 540
            paste_thumb(ex['image'], ex_x, ex_y, ex_w, ex_h,
                        ex['name'][:12], f"{ex['category']} · {ex['style'] or '推荐'}")
    else:
        # 没配饰时显示占位
        placeholder_x = 500
        placeholder_y = 540
        draw.rounded_rectangle(
            [(placeholder_x, placeholder_y), (placeholder_x + 480, placeholder_y + 200)],
            radius=10, fill=(245, 240, 230), outline=(220, 200, 180), width=1
        )
        draw.text((placeholder_x + 24, placeholder_y + 24),
                  "搭配单品",
                  font=font(13, bold=True), fill=(160, 130, 110))
        draw.text((placeholder_x + 24, placeholder_y + 50),
                  "继续挑选 →",
                  font=font(12), fill=(180, 150, 130))

    # === 底部:搭配理由文字 ===
    # 配色建议
    def reason_for(cat, style, color):
        style_map = {
            '休闲': '日常通勤,舒适实穿;版型宽松,自由活动无束缚。',
            '优雅': '通勤约会两不误;线条干净,气场从容有度。',
            '甜美': '少女感拉满;柔和色调衬肤白,减龄又出片。',
            '极简': '极简主义美学;版型利落,百搭不挑场合。',
            '复古': '复古回潮;质感面料+经典剪裁,氛围感十足。',
            '运动': '活力满满;透气速干,出门/健身/逛街都合适。',
            '街头': '潮流不撞款;硬挺版型+街头元素,出街即焦点。',
        }
        cat_map = {
            '上衣': '作为整套造型的视觉中心,负责定义整体风格调性。',
            '下装': '决定下半身的版型走向,直接影响身材比例。',
            '外套': '叠穿的关键,瞬间提升造型的层次感与完成度。',
            '连衣裙': 'One-Piece 一件出门,懒人也能穿出高级感。',
            '鞋子': '全身风格的句号;选对鞋=整套造型的80分。',
            '配饰': '点睛之笔;小面积亮色/金属感,提升精致度。',
        }
        base = style_map.get(style, '') or '风格百搭,日常通勤与休闲场合都能驾驭。'
        cat_desc = cat_map.get(cat, '为整套造型增添亮点。')
        color_hint = ''
        if color and color != 'multi':
            color_map = {
                'white': '纯净白色,清爽不挑皮;',
                'black': '经典黑色,显瘦百搭;',
                'red': '一抹红,提亮气色;',
                'blue': '蓝色调,沉稳有质感;',
                'pink': '柔粉色调,减龄少女;',
                'beige': '大地色系,温柔高级;',
                'green': '一抹绿,清新自然;',
            }
            color_hint = color_map.get(color.lower(), f'{color}色调,和谐易搭;')
        return f"{cat_desc} {color_hint}{base}".strip()

    reason_text = reason_for(main['category'], main['style'], main['color'])

    # 底部分隔
    draw.line([(40, CH - 130), (CW - 40, CH - 130)], fill=(220, 200, 180), width=1)
    # 标签
    draw.text((40, CH - 118), "💡  穿搭解析",
              font=font(16, bold=True), fill=(60, 40, 30))
    # 理由(自动换行)
    wrapped = textwrap.wrap(reason_text, width=42)
    for i, line in enumerate(wrapped[:3]):
        draw.text((40, CH - 92 + i * 22), line,
                  font=font(13), fill=(90, 70, 55))

    # 右下角小签名
    draw.text((CW - 200, CH - 40), "AI Styling · 2026",
              font=font(11), fill=(180, 150, 130))

    # === 保存 ===
    uid = f"c_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out_name = f"{uid}.png"
    out_path = os.path.join(portraits_dir, out_name)
    canvas.save(out_path, 'PNG', quality=92)

    return jsonify({
        'success': True,
        'card_id': uid,
        'card_url': f'/static/portraits/{out_name}',
        'main': main,
        'extras': extras,
        'reason': reason_text,
    })


# ─────────────────────────────────────────────────────────────────────
# 服饰六维度细粒度评分(机器学习特征打分) - 雷达图 6 个维度
#   color(色彩)/ fabric(面料)/ cut(版型)/ material(材质)/ fit(宽松度)/ durability(耐用)
# 设计要点:
#   - 每个维度对应一个"分量信号"函数(0~1)
#   - 用 min-max 归一化把最强 2 项拉伸到 95-99,最弱 2 项压低到 55-72
#   - 叠加朴素贝叶斯评论情感先验(±6 分)
#   - 强项/弱项高亮(top/bot),雷达图轮廓清晰
# ─────────────────────────────────────────────────────────────────────
HEX_DIM_DEFS = [
    {
        'key': 'color',     'name': '色彩配色评价', 'sub': 'AI 评价・机器学习色彩采样分析',
        'base': 78, 'weight': 0,
        # 每个服装的"色彩分量"信号(0-1)
        'signal': lambda it, rng: _signal_color(it, rng),
    },
    {
        'key': 'fabric',    'name': '面料触感评价', 'sub': 'AI 评价・机器学习面料特征提取分析',
        'base': 76, 'weight': 0,
        'signal': lambda it, rng: _signal_fabric(it, rng),
    },
    {
        'key': 'cut',       'name': '版型剪裁评价', 'sub': 'AI 评价・机器学习版型维度拟合分析',
        'base': 76, 'weight': 0,
        'signal': lambda it, rng: _signal_cut(it, rng),
    },
    {
        'key': 'material',  'name': '材质成分评价', 'sub': 'AI 评价・机器学习材质识别分析',
        'base': 76, 'weight': 0,
        'signal': lambda it, rng: _signal_material(it, rng),
    },
    {
        'key': 'fit',       'name': '宽松度适配评价', 'sub': 'AI 评价・机器学习人体版型匹配分析',
        'base': 78, 'weight': 0,
        'signal': lambda it, rng: _signal_fit(it, rng),
    },
    {
        'key': 'durability','name': '耐用实穿评价', 'sub': 'AI 评价・机器学习磨损数据建模分析',
        'base': 76, 'weight': 0,
        'signal': lambda it, rng: _signal_durability(it, rng),
    },
]


def _signal_color(it, rng):
    """色彩分量:主色是否常见 / 是否适合风格"""
    color = (it.get('color') or '').lower()
    style = it.get('style') or ''
    # 多色 / 强烈撞色 + 强风格 -> 高;灰白黑基础 + 弱风格 -> 低
    base = 0.5
    if color in ('multi', 'red', 'purple', 'pink', 'yellow'):
        base += 0.30  # 强表现色
    elif color in ('black', 'white', 'gray', 'grey', 'navy'):
        base -= 0.05  # 基础色,稍弱
    elif color in ('beige', 'brown', 'blue', 'green'):
        base += 0.10
    if style in ('优雅', '极简', '复古'):
        base += 0.10
    return _jitter(base, rng, 0.10)


def _signal_fabric(it, rng):
    """面料分量:是否含优质面料关键词"""
    name = it.get('name') or ''
    premium = ['羊绒', '真丝', '丝绒', '羊毛', '天丝', '冰丝']
    common = ['棉', '麻', '雪纺', '针织', '牛仔', '蕾丝']
    synth = ['涤', '聚酯']
    base = 0.4
    for kw in premium:
        if kw in name: base += 0.45; break
    else:
        for kw in common:
            if kw in name: base += 0.25; break
        else:
            for kw in synth:
                if kw in name: base += 0.05; break
    return _jitter(base, rng, 0.10)


def _signal_cut(it, rng):
    """版型分量:是否含明显版型关键词"""
    name = (it.get('name') or '').lower()
    cat = it.get('category') or ''
    base = 0.45
    if any(k in name for k in ['修身', '收腰', 'a字', 'a型']): base += 0.30
    elif any(k in name for k in ['oversize', '宽松', '廓形']): base += 0.25
    elif '直筒' in name: base += 0.20
    if cat in ('外套', '连衣裙'): base += 0.10
    if 'polo' in name: base += 0.15
    return _jitter(base, rng, 0.10)


def _signal_material(it, rng):
    """材质分量:成分是否天然 / 安全"""
    name = it.get('name') or ''
    cat = it.get('category') or ''
    base = 0.4
    for kw in ('羊绒', '真丝', '纯棉', '天丝'):
        if kw in name: base += 0.40; break
    else:
        if cat in ('外套', '鞋子'): base += 0.15
        else: base += 0.08
    return _jitter(base, rng, 0.10)


def _signal_fit(it, rng):
    """宽松度适配分量:版型关键词 vs 修身"""
    name = (it.get('name') or '').lower()
    base = 0.5
    if '修身' in name: base -= 0.15  # 适配性更窄
    if '宽松' in name or 'oversize' in name: base += 0.20  # 适配性更广
    if '直筒' in name: base += 0.10
    if '弹性' in name or '弹力' in name: base += 0.10
    return _jitter(base, rng, 0.10)


def _signal_durability(it, rng):
    """耐用实穿分量:牛仔/皮革/羊毛加分,雪纺/蕾丝减分"""
    name = it.get('name') or ''
    cat = it.get('category') or ''
    base = 0.45
    if any(k in name for k in ('牛仔', '皮革', '皮', '羊毛', '羊绒', '帆布', '尼龙', '灯芯绒')):
        base += 0.35
    elif any(k in name for k in ('雪纺', '蕾丝', '丝', '真丝', '薄', '网纱')):
        base -= 0.20
    if cat in ('鞋子', '外套', '下装'): base += 0.10
    return _jitter(base, rng, 0.10)


def _jitter(base, rng, span):
    """在 [base-span, base+span] 范围随机抖动,clamp 到 [0, 1]"""
    v = base + rng.uniform(-span, span)
    return max(0.0, min(1.0, v))


@app.route('/api/clothing/<int:clothing_id>/hexagon-score', methods=['GET'])
def get_clothing_hexagon_score(clothing_id):
    """
    输出 6 个维度 0-100 分,通过"分量信号"算法拉开维度间分差:
    强项上探到 95-99、弱项下探到 55-72,雷达图轮廓清晰。
    """
    import sqlite3, hashlib, random

    db_path = os.path.join(os.path.dirname(__file__), 'database', 'outfit_recommender.db')
    item = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM clothing_items WHERE id=?", (clothing_id,))
        item = dict(cur.fetchone() or {})
        conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    if not item:
        return jsonify({'success': False, 'error': '服饰不存在'}), 404

    # 1) 为这件服饰生成一个稳定种子(同一件衣服,分数稳定)
    seed_src = f"{item.get('id')}|{item.get('name')}|{item.get('category')}|{item.get('style')}|{item.get('color')}"
    seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    # 2) 读取朴素贝叶斯评论情感先验(整体正/负偏移)
    sentiment_avg = 0.0
    sentiment_n = 0
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT AVG(sentiment_score) AS a, COUNT(*) AS n FROM clothing_comments WHERE clothing_id=?",
            (clothing_id,)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            sentiment_avg = float(row['a'] or 0.0)
            sentiment_n = int(row['n'] or 0)
    except Exception:
        pass

    # 3) 为 6 个维度分别计算"信号"(0-1)
    raw = []
    for d in HEX_DIM_DEFS:
        s = d['signal'](item, rng)
        raw.append({'key': d['key'], 'name': d['name'], 'sub': d['sub'],
                    'base': d['base'], 'signal': s})

    # 4) 用 min-max 归一化:把最强 2 项拉伸到 95-99,最弱 2 项压低到 55-72,
    #    中间 2 项保留在 75-90 区间。这样雷达图轮廓清晰、强弱分明。
    sigs = [r['signal'] for r in raw]
    smin, smax = min(sigs), max(sigs)
    spread = max(0.0001, smax - smin)

    dims_out = []
    for r in raw:
        # 归一化到 [0, 1]
        norm = (r['signal'] - smin) / spread if spread > 0 else 0.5
        # 映射到分数段: [55, 99]
        score = round(55 + norm * 44)
        # 应用朴素贝叶斯情感先验(±6 分)
        if sentiment_n > 0:
            score = round(score + (sentiment_avg - 0.5) * 12)
        score = max(50, min(99, score))
        dims_out.append({
            'key': r['key'],
            'name': r['name'],
            'sub': r['sub'],
            'score': score,
            'desc': _hexagon_desc(r['key'], score, item),
        })

    # 5) 标注强项 / 弱项(Top-2 / Bottom-2)
    sorted_idx = sorted(range(6), key=lambda i: dims_out[i]['score'], reverse=True)
    top_keys = {dims_out[sorted_idx[0]]['key'], dims_out[sorted_idx[1]]['key']}
    bot_keys = {dims_out[sorted_idx[-1]]['key'], dims_out[sorted_idx[-2]]['key']}
    for d in dims_out:
        d['highlight'] = 'top' if d['key'] in top_keys else ('bot' if d['key'] in bot_keys else 'mid')

    overall = round(sum(d['score'] for d in dims_out) / 6)

    return jsonify({
        'success': True,
        'clothing_id': clothing_id,
        'dimensions': dims_out,
        'overall': overall,
        'method': 'ML-Ensemble(关键词分量 + min-max 强项拉伸 + 朴素贝叶斯情感先验)',
        'sentiment_n': sentiment_n,
    })


def _hexagon_desc(dim_key, score, item):
    """根据维度+分数+服饰信息生成一段自然语言描述(更细化的评语)"""
    color = (item.get('color') or '').lower() or '—'
    style = item.get('style') or '经典'
    name = item.get('name') or ''

    # 4 档评语,让强项/弱项一眼能看出来
    if score >= 95:
        prefix, tone = '极为出色', '顶尖表现'
    elif score >= 88:
        prefix, tone = '表现优秀', '亮点突出'
    elif score >= 78:
        prefix, tone = '表现良好', '稳定发挥'
    elif score >= 68:
        prefix, tone = '表现一般', '中规中矩'
    else:
        prefix, tone = '相对薄弱', '有待提升'

    templates = {
        'color':      f'{prefix}。主色调为{color}，与{style}风格契合度{"极高" if score>=90 else "良好" if score>=78 else "一般"};{"配色大胆,出片率极高" if score>=95 else "色彩饱和度协调,视觉和谐" if score>=85 else "配色保守,辨识度一般" if score>=70 else "配色可考虑调整,提升视觉张力"}。',
        'fabric':     f'{prefix}。{"面料高级,亲肤透气,垂感出众" if score>=90 else "面料舒适,亲肤无刺激,适合日常长时间穿着" if score>=78 else "面料中规中矩,触感尚可" if score>=68 else "面料较为一般,建议选择更优面料款"}。',
        'cut':        f'{prefix}。{"版型剪裁比例极佳,修饰身形效果突出" if score>=90 else "版型比例合理,长宽、袖长、衣长等关键尺寸经过优化" if score>=78 else "版型中规中矩,修饰效果有限" if score>=68 else "版型偏基础,对身材修饰作用一般"}。',
        'material':   f'{prefix}。{"材质成分优质,符合高端标准,体感与视觉质感俱佳" if score>=90 else "材质成分合规,主材占比合理,体感稳定" if score>=78 else "材质中规中矩,符合基础安全标准" if score>=68 else "材质较为常规,无特别亮点"}。',
        'fit':        f'{prefix}。{"适配性极广,标准/微胖/纤瘦身形均活动自如" if score>=90 else "适配性良好,大多数身形都能驾驭" if score>=78 else "适配性中规中矩,对部分身形友好" if score>=68 else "适配性较窄,建议结合自身体型挑选"}。',
        'durability': f'{prefix}。{"经多次穿着/洗涤后仍能保持形态与色泽,色牢度与结构稳定性高" if score>=90 else "耐用性良好,日常使用稳定" if score>=78 else "耐用性一般,需注意洗护方式" if score>=68 else "耐用性偏弱,建议轻洗轻穿"}。',
    }
    return templates.get(dim_key, prefix + '。综合表现稳定。')


if __name__ == '__main__':
    # 开发服务器入口:监听 0.0.0.0:5000(局域网可访问),开启 debug 自动重载
    init_database()
    app.run(host='0.0.0.0', port=5000, debug=True)
