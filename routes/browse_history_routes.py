"""
============================================================
浏览历史路由(browse_history_routes.py)
============================================================
【作用】/api/browse-history 前缀下的接口
  - POST  /api/browse-history              记录一次浏览
  - GET   /api/browse-history/<user_id>    查某用户历史
  - DELETE /api/browse-history/<id>        单条删除
  - POST  /api/browse-history/clear/<uid>  一键清空
【数据表】browse_history
============================================================
"""

# ── Flask ──
from flask import Blueprint, request, jsonify

# ── 业务服务 ──
from services.browse_history_service import BrowseHistoryService

# 蓝图 + service
browse_bp = Blueprint('browse_history', __name__, url_prefix='/api/browse-history')
browse_service = BrowseHistoryService()


# ── POST /api/browse-history ── 记录一次浏览 ──
@browse_bp.route('', methods=['POST'])
def record_browse():
    """
    记录一次浏览事件
    Body:{"user_id":1, "item_type":"outfit", "item_id":null, "title":"...", "image_url":"/static/...", "meta":{...}}
    """
    data      = request.json or {}
    user_id   = data.get('user_id')
    item_type = data.get('item_type')      # 'outfit' | 'clothing'
    result    = browse_service.record(
        user_id=user_id,
        item_type=item_type,
        item_id=data.get('item_id'),
        title=data.get('title'),
        image_url=data.get('image_url'),
        meta=data.get('meta')
    )
    status = 200 if result.get('success') else 400
    return jsonify(result), status


# ── GET /api/browse-history/<user_id> ── 查历史 ──
@browse_bp.route('/<int:user_id>', methods=['GET'])
def list_browse(user_id):
    """取某用户最近 N 条浏览记录(默认 50)"""
    limit  = request.args.get('limit', 50, type=int)
    result = browse_service.list_history(user_id, limit)
    return jsonify(result)


# ── DELETE /api/browse-history/<history_id> ── 单条删除 ──
@browse_bp.route('/<int:history_id>', methods=['DELETE'])
def delete_browse(history_id):
    """按主键 ID 删除一条"""
    result = browse_service.delete(history_id)
    status = 200 if result.get('success') else 404
    return jsonify(result), status


# ── POST /api/browse-history/clear/<user_id> ── 一键清空 ──
@browse_bp.route('/clear/<int:user_id>', methods=['POST'])
def clear_browse(user_id):
    """清空某用户所有历史(返回删除条数)"""
    result = browse_service.clear(user_id)
    return jsonify(result)
