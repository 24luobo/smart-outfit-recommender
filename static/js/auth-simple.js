// 简化版认证功能 - 带localStorage状态保持和身材数据
let currentUser = null;
let isGuest = false;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Auth JS loaded');
    
    // 从localStorage读取登录状态
    loadAuthState();
    
    // 不再自动显示登录弹窗，只更新UI
    updateAuthUI();
    // 检查是否需要跳转到身材录入
    checkAndRedirectToBodyShape();
});

// 从localStorage加载状态
function loadAuthState() {
    const savedUser = localStorage.getItem('currentUser');
    const savedIsGuest = localStorage.getItem('isGuest');
    
    if (savedUser) {
        currentUser = savedUser;
        isGuest = savedIsGuest === 'true';
        console.log('Loaded auth state from localStorage:', { currentUser, isGuest });
    }
}

// 保存状态到localStorage
function saveAuthState() {
    localStorage.setItem('currentUser', currentUser || '');
    localStorage.setItem('isGuest', isGuest ? 'true' : 'false');
    console.log('Saved auth state to localStorage:', { currentUser, isGuest });
}

// 保存用户基本信息
function saveUserProfile(data) {
    if (currentUser && !isGuest) {
        localStorage.setItem(`user_profile_${currentUser}`, JSON.stringify(data));
        console.log('User profile saved:', data);
    }
}

// 读取用户基本信息
function getUserProfile() {
    if (currentUser && !isGuest) {
        const data = localStorage.getItem(`user_profile_${currentUser}`);
        if (data) {
            return JSON.parse(data);
        }
    }
    return null;
}

// 保存身材数据
function saveBodyShapeData(data) {
    if (currentUser) {
        if (isGuest) {
            // 游客：临时保存
            localStorage.setItem('guest_body_shape', JSON.stringify(data));
        } else {
            // 登录用户：按账号保存
            localStorage.setItem(`body_shape_${currentUser}`, JSON.stringify(data));
        }
        console.log('Body shape data saved:', data);
    }
}

// 读取身材数据
function getBodyShapeData() {
    if (currentUser) {
        let data = null;
        if (isGuest) {
            data = localStorage.getItem('guest_body_shape');
        } else {
            data = localStorage.getItem(`body_shape_${currentUser}`);
        }
        if (data) {
            return JSON.parse(data);
        }
    }
    return null;
}

// 检查是否有身材数据
function hasBodyShapeData() {
    return getBodyShapeData() !== null;
}

// 检查并跳转到身材录入（如果需要）
function checkAndRedirectToBodyShape() {
    // 只在获取推荐页面检查
    const currentPath = window.location.pathname;
    if (currentPath.includes('recommendation') && currentUser && !hasBodyShapeData() && !isGuest) {
        console.log('No body shape data, redirecting to body shape page');
        // 显示提示，跳转到身材信息页面
        setTimeout(() => {
            alert('请先填写身材信息');
            window.location.href = '/body-shape';
        }, 300);
    }
}

// 更新认证UI
function updateAuthUI() {
    console.log('Updating auth UI:', { currentUser, isGuest });
    
    // 更新游客提示条
    const banner = document.getElementById('guest-banner');
    if (banner) {
        if (isGuest) {
            banner.style.display = 'block';
            banner.style.marginTop = '50px';
        } else {
            banner.style.display = 'none';
            banner.style.marginTop = '0';
        }
    }
    
    // 更新导航栏
    const navAuth = document.getElementById('nav-auth');
    if (navAuth) {
        if (currentUser && !isGuest) {
            const profile = getUserProfile();
            let welcomeText = `欢迎，${currentUser}`;
            
            if (profile) {
                const skinToneMap = { 'fair': '冷白皮', 'medium': '暖黄皮', 'dark': '小麦色', 'deep': '深肤色' };
                const styleMap = { 'casual': '休闲', 'formal': '商务', 'sporty': '运动', 'elegant': '优雅', 'trendy': '潮流', 'minimal': '极简' };
                const sceneMap = { 'work': '职场', 'casual': '日常', 'dating': '约会', 'party': '派对', 'travel': '旅行', 'sport': '运动' };
                
                const skinTone = skinToneMap[profile.skinTone] || '';
                const style = styleMap[profile.style] || '';
                const scene = sceneMap[profile.scene] || '';
                
                if (skinTone || style || scene) {
                    const extraInfo = [skinTone, style, scene].filter(x => x).join(' · ');
                    welcomeText = `欢迎，${currentUser} (${extraInfo})`;
                }
            }
            
            navAuth.innerHTML = `
                <span style="color: #6B6560; margin-right: 10px;">${welcomeText}</span>
                <button onclick="logoutUser()" class="btn btn-secondary btn-small">退出</button>
            `;
        } else if (isGuest) {
            navAuth.innerHTML = `
                <span style="color: #8A847E; margin-right: 10px;">游客模式</span>
                <button onclick="showLoginModal()" class="btn btn-secondary btn-small">登录</button>
            `;
        } else {
            navAuth.innerHTML = `
                <button onclick="showLoginModal()" class="btn btn-secondary btn-small">登录</button>
                <button onclick="showRegisterModal()" class="btn btn-primary btn-small">注册</button>
            `;
        }
    }
    
    // 关闭弹窗
    const modal = document.getElementById('auth-modal');
    if (modal && currentUser) {
        modal.style.display = 'none';
    }
}

function closeAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function continueAsGuest() {
    console.log('Guest mode clicked');
    
    isGuest = true;
    currentUser = 'guest_' + Date.now();
    
    saveAuthState();
    closeAuthModal();
    updateAuthUI();
    
    console.log('Guest mode setup complete');
}

function showLoginModal() {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    if (loginForm) loginForm.style.display = 'block';
    if (registerForm) registerForm.style.display = 'none';
    
    const modal = document.getElementById('auth-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

function showRegisterModal() {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    if (loginForm) loginForm.style.display = 'none';
    if (registerForm) registerForm.style.display = 'block';
    
    const modal = document.getElementById('auth-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

// 简单的登录函数（前端模拟）
function loginUser() {
    const username = document.getElementById('login-username')?.value;
    const password = document.getElementById('login-password')?.value;
    
    console.log('Login clicked:', { username, password });
    
    if (!username || !password) {
        alert('请填写用户名和密码');
        return;
    }
    
    isGuest = false;
    currentUser = username;
    
    saveAuthState();
    closeAuthModal();
    updateAuthUI();
    
    // 检查是否需要跳转到身材录入
    setTimeout(() => {
        checkAndRedirectToBodyShape();
    }, 200);
    
    console.log('Login successful');
    alert('登录成功！');
}

// 简单的注册函数（前端模拟）
function registerUser() {
    const username = document.getElementById('register-username')?.value;
    const password = document.getElementById('register-password')?.value;
    const confirmPassword = document.getElementById('register-confirm-password')?.value;
    const skinTone = document.getElementById('register-skin-tone')?.value;
    const style = document.getElementById('register-style')?.value;
    const scene = document.getElementById('register-scene')?.value;
    
    console.log('Register clicked:', { username, password, confirmPassword, skinTone, style, scene });
    
    if (!username || !password || !confirmPassword) {
        alert('请填写用户名和密码');
        return;
    }
    
    if (password !== confirmPassword) {
        alert('两次输入的密码不一致');
        return;
    }
    
    isGuest = false;
    currentUser = username;
    
    // 保存用户基本信息
    if (skinTone || style || scene) {
        saveUserProfile({ skinTone, style, scene });
    }
    
    saveAuthState();
    closeAuthModal();
    updateAuthUI();
    
    // 注册成功后，跳转到身材录入页面
    setTimeout(() => {
        alert('注册成功！请先填写身材信息');
        window.location.href = '/profile';
    }, 300);
    
    console.log('Register successful');
}

function logoutUser() {
    currentUser = null;
    isGuest = false;
    
    saveAuthState();
    updateAuthUI();
    
    // 重新显示登录弹窗
    const modal = document.getElementById('auth-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
    
    console.log('Logged out');
}

// 点击弹窗外部关闭
document.addEventListener('click', function(e) {
    const modal = document.getElementById('auth-modal');
    if (modal && e.target === modal) {
        closeAuthModal();
    }
});
