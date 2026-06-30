"""
============================================================
身材信息路由(body_shape_routes.py)
============================================================
【作用】/api/body-shape 前缀下的接口
  - GET   /api/body-shape                       取身材信息
  - POST  /api/body-shape                       保存身材信息
  - GET   /api/body-shape/calculate/<item_id>   计算身材-服饰匹配度
【数据表】body_shape
============================================================
"""

# ── Flask ──
from flask import Blueprint, request, jsonify

# ── 数据库访问层(身材页直接用 DB,没走 service) ──
from database.models import DatabaseManager

# 蓝图 + DB 实例
body_shape_bp = Blueprint('body_shape', __name__, url_prefix='/api/body-shape')
db = DatabaseManager()


# ── GET /api/body-shape ── 取身材信息 ──
@body_shape_bp.route('', methods=['GET'])
def get_body_shape():
    """取用户身材信息(简化版,user_id 写死 1)"""
    try:
        user_id    = 1
        body_shape = db.get_body_shape(user_id)
        return jsonify(body_shape)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── POST /api/body-shape ── 保存身材信息 ──
@body_shape_bp.route('', methods=['POST'])
def save_body_shape():
    """
    保存身材信息
    Body:{
        "height":170, "weight":55, "shoulder":40, "waist":65, "hip":88,
        "leg":85, "skin_tone":"medium", "body_type":"slim"
    }
    """
    try:
        data    = request.get_json()
        user_id = 1

        # 提取 8 个字段(都是可选的)
        height     = data.get('height')
        weight     = data.get('weight')
        shoulder   = data.get('shoulder')
        waist      = data.get('waist')
        hip        = data.get('hip')
        leg        = data.get('leg')
        skin_tone  = data.get('skin_tone')
        body_type  = data.get('body_type')

        # 字符串转 float(防止 JS 传过来是字符串)
        if height:   height   = float(height)
        if weight:   weight   = float(weight)
        if shoulder: shoulder = float(shoulder)
        if waist:    waist    = float(waist)
        if hip:      hip      = float(hip)
        if leg:      leg      = float(leg)

        # 调 DB 层 upsert
        success = db.update_body_shape(
            user_id=user_id,
            height=height, weight=weight,
            shoulder=shoulder, waist=waist, hip=hip, leg=leg,
            skin_tone=skin_tone, body_type=body_type
        )

        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── GET /api/body-shape/calculate/<item_id> ── 身材-服饰匹配度 ──
@body_shape_bp.route('/calculate/<int:item_id>', methods=['GET'])
def calculate_body_shape_match(item_id: int):
    """
    计算某件衣服对当前用户身材的匹配度(0~1)
    演示版:固定返回 0.85 + 命中项明细
    """
    try:
        user_id    = 1
        body_shape = db.get_body_shape(user_id)

        if not body_shape:
            return jsonify({'score': 0.7, 'details': '使用默认身材适配度'})

        score   = 0.85
        details = {
            'body_shape':       body_shape.get('body_type', '未知'),
            'height_match':     True,
            'weight_match':     True,
            'skin_tone_match':  True
        }

        return jsonify({'score': score, 'details': details})
    except Exception as e:
        return jsonify({'score': 0.7, 'details': {'error': str(e)}}), 500
