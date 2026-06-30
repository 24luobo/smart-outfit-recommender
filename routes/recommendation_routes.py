"""
============================================================
推荐路由(recommendation_routes.py) - 最核心的接口
============================================================
【作用】/api/recommendations 前缀下的接口
  - POST  /api/recommendations              朴素贝叶斯 + K-Means 融合 → Top-5 穿搭
  - POST  /api/recommendations/train        重训所有模型
  - GET   /api/recommendations/history/<id> 查历史推荐
  - PUT   /api/recommendations/feedback/<id> 提交点赞/踩
============================================================
"""

# ── Flask ──
from flask import Blueprint, request, jsonify

# ── 业务服务(朴素贝叶斯 + K-Means 融合的主流程) ──
from services.recommendation_service import RecommendationService

# 蓝图 + service
recommendation_bp = Blueprint('recommendation', __name__, url_prefix='/api/recommendations')
recommendation_service = RecommendationService()


# ── POST /api/recommendations ── 朴素贝叶斯 + K-Means 融合推荐 ──
@recommendation_bp.route('', methods=['POST'])
def get_recommendations():
    """
    朴素贝叶斯 + K-Means 融合的穿搭推荐接口
    Body:{"user_id":1, "temperature":22, "weather_condition":"sunny", "season":"spring"}

    流程:
      ① 朴素贝叶斯 筛候选 → ② K-Means 风格聚类 → ③ 落库 recommendation_history

    返回:5 套穿搭方案
    """
    data = request.json
    required_fields = ['user_id', 'temperature', 'weather_condition', 'season']

    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error':   'Missing required fields: user_id, temperature, weather_condition, and season are required'
        }), 400

    result = recommendation_service.get_recommendations(
        user_id=data['user_id'],
        temperature=data['temperature'],
        weather_condition=data['weather_condition'],
        season=data['season']
    )

    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


# ── POST /api/recommendations/train ── 重训所有模型 ──
@recommendation_bp.route('/train', methods=['POST'])
def train_models():
    """
    重训所有模型(添加新服饰后,或修改了训练数据时调用)
    返回:{'success': True/False, 'message': '...'}
    """
    result = recommendation_service.train_models()
    if result['success']:
        return jsonify(result)
    return jsonify(result), 500


# ── GET /api/recommendations/history/<user_id> ── 查历史 ──
@recommendation_bp.route('/history/<int:user_id>', methods=['GET'])
def get_recommendation_history(user_id):
    """查某用户最近的推荐历史(limit 由 query string 指定,默认 10)"""
    limit = request.args.get('limit', 10, type=int)
    result = recommendation_service.get_recommendation_history(user_id, limit)
    return jsonify(result)


# ── PUT /api/recommendations/feedback/<history_id> ── 用户反馈 ──
@recommendation_bp.route('/feedback/<int:history_id>', methods=['PUT'])
def submit_feedback(history_id):
    """
    用户对一次推荐点赞/踩,反馈写到 recommendation_history.feedback
    Body:{"feedback": 1}  1=赞,0=踩
    """
    data = request.json
    if 'feedback' not in data:
        return jsonify({'success': False, 'error': 'Missing feedback'}), 400

    result = recommendation_service.submit_feedback(history_id, data['feedback'])
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400
