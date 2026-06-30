"""
============================================================
收藏路由(favorite_routes.py)
============================================================
【作用】/api/favorites 前缀下的接口
  - POST  /api/favorites                 收藏(带去重)
  - GET   /api/favorites/<user_id>       列出某用户所有收藏
  - DELETE /api/favorites/<favorite_id>  按 ID 删除
  - POST  /api/favorites/remove          按签名删除(前端只知 source+key 时)
============================================================
"""

# ── Flask ──
from flask import Blueprint, request, jsonify

# ── 业务服务 ──
from services.favorite_service import FavoriteService

# 蓝图 + service
favorite_bp = Blueprint('favorite', __name__, url_prefix='/api/favorites')
favorite_service = FavoriteService()


# ── POST /api/favorites ── 收藏(带去重) ──
@favorite_bp.route('', methods=['POST'])
def create_favorite():
    """
    收藏一个穿搭方案
    Body:{"user_id":1, "outfit_combo":[1,2,3], "weather_context":{"temperature":22,"season":"spring"}}
    """
    data = request.json
    required_fields = ['user_id', 'outfit_combo']

    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error':   'Missing required fields: user_id and outfit_combo are required'
        }), 400

    result = favorite_service.create_favorite(
        user_id=data['user_id'],
        outfit_combo=data['outfit_combo'],
        weather_context=data.get('weather_context')
    )

    if result['success']:
        return jsonify(result), 201
    return jsonify(result), 400


# ── GET /api/favorites/<user_id> ── 列收藏 ──
@favorite_bp.route('/<int:user_id>', methods=['GET'])
def get_user_favorites(user_id):
    """收藏页"我的收藏"模块用"""
    result = favorite_service.get_user_favorites(user_id)
    return jsonify(result)


# ── DELETE /api/favorites/<favorite_id> ── 按 ID 删除 ──
@favorite_bp.route('/<int:favorite_id>', methods=['DELETE'])
def delete_favorite(favorite_id):
    """按主键 ID 删除(管理用)"""
    result = favorite_service.delete_favorite(favorite_id)
    if result['success']:
        return jsonify(result)
    return jsonify(result), 404


# ── POST /api/favorites/remove ── 按签名删除 ──
@favorite_bp.route('/remove', methods=['POST'])
def remove_favorite_by_signature():
    """
    通过签名取消收藏(前端只知道 source+key 时使用)
    Body:{"user_id":1, "signature":"outfit:xxx:yyy"}
    """
    data = request.json or {}
    user_id   = data.get('user_id')
    signature = data.get('signature')
    if not user_id or not signature:
        return jsonify({'success': False, 'error': 'user_id 和 signature 必填'}), 400
    result = favorite_service.delete_favorite_by_signature(user_id, signature)
    status = 200 if result['success'] else 404
    return jsonify(result), status
