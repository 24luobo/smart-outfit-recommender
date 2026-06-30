"""
============================================================
认证路由(auth_routes.py)
============================================================
【作用】账号体系 + 游客模式
  - POST  /api/auth/register   注册新账号
  - POST  /api/auth/login      登录
  - POST  /api/auth/guest      创建/获取游客(幂等,cookie 保留 1 年)
  - POST  /api/auth/logout     退出
  - GET   /api/auth/status     查询当前登录状态
【亮点】游客模式三重兜底:session → cookie → 新建,保证零门槛体验
============================================================
"""

# ── Flask ──
from flask import Blueprint, request, jsonify, session

# ── 数据库访问层 ──
from database.models import DatabaseManager

# ── 密码哈希 ──
import hashlib

# 蓝图 + DB 实例
auth_bp = Blueprint('auth', __name__)
db = DatabaseManager()


# ── 工具:SHA256 哈希密码(生产建议 bcrypt) ──
def hash_password(password: str) -> str:
    """简单 SHA256 哈希;生产环境建议用 bcrypt/argon2"""
    return hashlib.sha256(password.encode()).hexdigest()


# ── POST /api/auth/register ── 注册 ──
@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    """
    注册新账号
    Body:{"username":"alice", "password":"123456"}
    """
    try:
        data     = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'}), 400

        # 1) 检查用户名是否已存在
        existing_user = db.get_user_by_username(username)
        if existing_user:
            return jsonify({'success': False, 'message': '用户名已存在'}), 400

        # 2) 哈希入库
        password_hash = hash_password(password)
        user_id       = db.create_user(
            height=None, skin_tone=None,
            username=username, password_hash=password_hash, is_guest=False
        )

        # 3) 写 session(后续请求免登录)
        session['user_id']  = user_id
        session['is_guest'] = False

        return jsonify({
            'success':   True,
            'message':   '注册成功',
            'user_id':   user_id,
            'username':  username
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ── POST /api/auth/login ── 登录 ──
@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    """
    登录
    Body:{"username":"alice", "password":"123456"}
    """
    try:
        data     = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'}), 400

        # 1) 验证密码哈希
        password_hash = hash_password(password)
        user          = db.verify_user(username, password_hash)

        if not user:
            return jsonify({'success': False, 'message': '用户名或密码错误'}), 401

        # 2) 写 session
        session['user_id']  = user['id']
        session['is_guest'] = False

        # 3) 检查是否已有身材数据(决定前端是否弹身材表单)
        body_shape = db.get_body_shape(user['id'])

        return jsonify({
            'success':       True,
            'message':       '登录成功',
            'user_id':       user['id'],
            'username':      username,
            'has_body_shape': body_shape is not None
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ── POST /api/auth/guest ── 游客模式(幂等) ──
@auth_bp.route('/api/auth/guest', methods=['POST'])
def guest():
    """
    游客模式(零门槛体验):
      1) 先看 session 是否有游客,有 → 直接复用
      2) 再看 cookie guest_uid,有 → 复用
      3) 都没有 → 新建一个游客,写 session + set 1 年 cookie
    """
    try:
        # 1) session 命中
        existing_id   = session.get('user_id')
        existing_user = db.get_user(existing_id) if existing_id else None
        if existing_user and session.get('is_guest'):
            return jsonify({
                'success':  True,
                'message':  '已进入游客模式',
                'user_id':  existing_user['id'],
                'is_guest': True
            })

        # 2) cookie 命中
        cookie_uid = request.cookies.get('guest_uid')
        if cookie_uid:
            try:
                cookie_user = db.get_user(int(cookie_uid))
                if cookie_user and cookie_user.get('is_guest'):
                    session['user_id']  = cookie_user['id']
                    session['is_guest'] = True
                    return jsonify({
                        'success':  True,
                        'message':  '已进入游客模式',
                        'user_id':  cookie_user['id'],
                        'is_guest': True
                    })
            except (ValueError, TypeError):
                pass

        # 3) 新建游客
        user_id              = db.create_guest_user()
        session['user_id']   = user_id
        session['is_guest']  = True

        resp = jsonify({
            'success':  True,
            'message':  '进入游客模式',
            'user_id':  user_id,
            'is_guest': True
        })
        # 1 年 cookie,让浏览器持久化
        resp.set_cookie('guest_uid', str(user_id), max_age=60 * 60 * 24 * 365, samesite='Lax')
        return resp
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ── POST /api/auth/logout ── 退出登录 ──
@auth_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    """清空 session(游客不会被真正删,下次还能复用)"""
    session.clear()
    return jsonify({'success': True, 'message': '已退出登录'})


# ── GET /api/auth/status ── 当前登录状态 ──
@auth_bp.route('/api/auth/status', methods=['GET'])
def auth_status():
    """
    供前端"我的"页面判断:
    - logged_in 是否登录
    - is_guest 是否游客
    - has_body_shape 是否有身材数据
    """
    user_id  = session.get('user_id')
    is_guest = session.get('is_guest', False)

    if not user_id:
        return jsonify({'logged_in': False, 'is_guest': False})

    user       = db.get_user(user_id)
    body_shape = db.get_body_shape(user_id) if user else None

    return jsonify({
        'logged_in':     True,
        'is_guest':      is_guest,
        'user_id':       user_id,
        'username':      user.get('username') if user else None,
        'has_body_shape': body_shape is not None
    })
