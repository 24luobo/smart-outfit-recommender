"""
============================================================
服饰路由(clothing_routes.py)
============================================================
【作用】/api/clothing 前缀下的接口
  - POST   /api/clothing                        新增一件服饰
  - GET    /api/clothing                        服饰库总览
  - GET    /api/clothing/<id>                   单件详情
  - GET    /api/clothing/category/<category>    按类目筛选
  - PUT    /api/clothing/<id>/cluster           写回聚类 ID
  - PUT    /api/clothing/<id>/features          写回特征向量
============================================================
"""

# ── Flask ──
from flask import Blueprint, request, jsonify

# ── 业务服务 ──
from services.clothing_service import ClothingService

# 蓝图 + service
clothing_bp = Blueprint('clothing', __name__, url_prefix='/api/clothing')
clothing_service = ClothingService()


# ── POST /api/clothing ── 新增服饰 ──
@clothing_bp.route('', methods=['POST'])
def create_clothing_item():
    """
    新增一件服饰(必填 name 和 category)
    Body:{
        "name":"白色T恤", "category":"上衣",
        "color":"white", "style":"casual", "season":"all",
        "suitable_temperature_min":18, "suitable_temperature_max":35,
        "image_path":"/static/images/xxx.png"
    }
    """
    data = request.json
    required_fields = ['name', 'category']

    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error':   'Missing required fields: name and category are required'
        }), 400

    result = clothing_service.create_clothing_item(
        name=data['name'],
        category=data['category'],
        color=data.get('color'),
        style=data.get('style'),
        season=data.get('season'),
        temp_min=data.get('suitable_temperature_min'),
        temp_max=data.get('suitable_temperature_max'),
        image_path=data.get('image_path'),
        feature_vector=data.get('feature_vector'),
        cluster_id=data.get('cluster_id')
    )

    if result['success']:
        return jsonify(result), 201
    return jsonify(result), 400


# ── GET /api/clothing/<id> ── 单件详情 ──
@clothing_bp.route('/<int:item_id>', methods=['GET'])
def get_clothing_item(item_id):
    """按 ID 取一件服饰(详情页用)"""
    result = clothing_service.get_clothing_item(item_id)
    if result['success']:
        return jsonify(result)
    return jsonify(result), 404


# ── GET /api/clothing ── 服饰库总览 ──
@clothing_bp.route('', methods=['GET'])
def get_all_clothing_items():
    """全量服饰列表(服饰库页用,前端要分页)"""
    result = clothing_service.get_all_clothing_items()
    return jsonify(result)


# ── GET /api/clothing/category/<category> ── 按类目筛选 ──
@clothing_bp.route('/category/<category>', methods=['GET'])
def get_clothing_by_category(category):
    """category:上衣 / 裤子 / 外套 / 连衣裙 / 裙子 / 鞋子 / 配饰"""
    result = clothing_service.get_clothing_by_category(category)
    return jsonify(result)


# ── PUT /api/clothing/<id>/cluster ── 写回聚类 ID ──
@clothing_bp.route('/<int:item_id>/cluster', methods=['PUT'])
def update_clothing_cluster(item_id):
    """KMeans 训练完后调用,把 cluster_id 写回 DB"""
    data = request.json
    if 'cluster_id' not in data:
        return jsonify({'success': False, 'error': 'Missing cluster_id'}), 400

    result = clothing_service.update_clothing_cluster(item_id, data['cluster_id'])
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


# ── PUT /api/clothing/<id>/features ── 写回特征向量 ──
@clothing_bp.route('/<int:item_id>/features', methods=['PUT'])
def update_clothing_features(item_id):
    """把 128 维图像特征写回 DB(冗余存储,避免每次实时重算)"""
    data = request.json
    if 'feature_vector' not in data:
        return jsonify({'success': False, 'error': 'Missing feature_vector'}), 400

    result = clothing_service.update_clothing_features(item_id, data['feature_vector'])
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400
