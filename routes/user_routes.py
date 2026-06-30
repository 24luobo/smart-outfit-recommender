"""
============================================================
用户路由(user_routes.py)
============================================================
【作用】/api/users 前缀下的所有 HTTP 接口
  - POST   /api/users        创建用户
  - GET    /api/users/<id>   查单个用户
  - PUT    /api/users/<id>   局部更新用户
  - GET    /api/users        全量用户(管理用)
【设计】薄路由层:只做"参数提取 + 必填校验 + 调 service",业务逻辑都在 services 层
============================================================
"""

# ── Flask 核心对象 ──
from flask import Blueprint, request, jsonify

# ── 业务服务 ──
from services.user_service import UserService

# 创建蓝图,所有接口自动加 /api/users 前缀
user_bp = Blueprint('user', __name__, url_prefix='/api/users')

# Service 单例(整个进程复用,避免重复建数据库连接)
user_service = UserService()


# ── POST /api/users ── 新建用户 ──
@user_bp.route('', methods=['POST'])
def create_user():
    """
    创建新用户(必填 height 和 skin_tone)
    Body 形如:{ "height":170, "skin_tone":"medium", "weight":60, "style_preference":"casual", "usual_scenes":["daily","work"] }
    """
    data = request.json
    required_fields = ['height', 'skin_tone']

    # 必填校验
    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error':   'Missing required fields: height and skin_tone are required'
        }), 400

    # 调 service 做事
    result = user_service.create_user(
        height=data['height'],
        skin_tone=data['skin_tone'],
        style_preference=data.get('style_preference'),  # 选填
        usual_scenes=data.get('usual_scenes'),
        weight=data.get('weight')
    )

    # 201 Created 表示资源已建立;400 表示参数/服务报错
    if result['success']:
        return jsonify(result), 201
    return jsonify(result), 400


# ── GET /api/users/<id> ── 查用户 ──
@user_bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """按 ID 取用户档案;找不到时 404"""
    result = user_service.get_user(user_id)
    if result['success']:
        return jsonify(result)
    return jsonify(result), 404


# ── PUT /api/users/<id> ── 更新用户 ──
@user_bp.route('/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """
    局部更新:Body 里只传要改的字段
    例:{"style_preference":"sporty"} → 只改这一项,其它保持不变
    """
    data = request.json
    result = user_service.update_user(user_id, **data)
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


# ── GET /api/users ── 全量用户(管理/调试用) ──
@user_bp.route('', methods=['GET'])
def get_all_users():
    """列出所有用户(管理后台/调试用)"""
    result = user_service.get_all_users()
    return jsonify(result)
