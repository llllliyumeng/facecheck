/* Tokens are exchanged once for an HttpOnly session; never persist them in browser storage. */
(() => {
  let enabled = false, pendingLogin = null;
  const nativeFetch = window.fetch.bind(window);
  function applyIdentity(identity, deployment) {
    const note = document.querySelector('.side-note');
    if (note) note.textContent = enabled ? `${deployment === 'cloud' ? 'HTTPS 云端部署模式' : '认证部署模式'} · ${identity.id} / ${identity.role}` : '本机演示模式 · 仅处理获得授权的测试素材';
    let logout = document.getElementById('authLogout');
    if (!logout && enabled) {
      logout = document.createElement('button'); logout.id = 'authLogout'; logout.className = 'button secondary'; logout.textContent = '退出登录';
      document.querySelector('.topbar').appendChild(logout);
      logout.onclick = async () => {await nativeFetch('/api/v1/auth/logout', {method: 'POST'}); window.location.reload();};
    }
    document.querySelectorAll('.nav button').forEach(b => {
      b.hidden = identity.role === 'auditor' && !['audit', 'security', 'requirements', 'benchmarks'].includes(b.dataset.view);
    });
    if (identity.role === 'auditor' && window.setView) window.setView('audit');
  }
  function login(deployment) {
    if (pendingLogin) return pendingLogin;
    pendingLogin = new Promise(resolve => {
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#122237ed;display:grid;place-items:center;padding:20px';
      const form = document.createElement('form'); form.className = 'card'; form.style.cssText = 'width:min(440px,100%);padding:30px';
      form.innerHTML = '<h2>登录 FaceCheck</h2><p>请输入管理员为你分配的访问令牌。令牌不会保存在浏览器存储中。</p><label for="accessToken">访问令牌</label><input id="accessToken" type="password" autocomplete="off" required maxlength="256" style="width:100%;margin:12px 0"><p role="alert" class="muted"></p><button class="button" type="submit">登录</button>';
      overlay.appendChild(form); document.body.appendChild(overlay);
      const input = form.querySelector('input'), message = form.querySelector('[role=alert]'), button = form.querySelector('button');
      form.onsubmit = async event => {
        event.preventDefault(); button.disabled = true; message.textContent = '';
        try {
          const response = await nativeFetch('/api/v1/auth/login', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({token: input.value})});
          input.value = '';
          const body = await response.json();
          if (!response.ok) throw new Error(response.status === 429 ? '尝试过于频繁，请一分钟后再试' : '登录失败，请核对访问令牌');
          overlay.remove(); applyIdentity(body, deployment); pendingLogin = null; resolve();
        } catch (error) {message.textContent = error.message;} finally {input.value = ''; button.disabled = false;}
      };
      input.focus();
    });
    return pendingLogin;
  }
  let deployment = 'local';
  const ready = (async () => {
    const response = await nativeFetch('/api/v1/auth/status');
    if (!response.ok) throw new Error('访问控制状态不可用');
    const status = await response.json(); enabled = status.enabled; deployment = status.deployment;
    if (!enabled) return;
    const me = await nativeFetch('/api/v1/auth/me');
    if (me.ok) applyIdentity(await me.json(), deployment); else await login(deployment);
  })();
  window.facecheckAuth = {
    ready,
    async fetch(url, options = {}) {
      await ready;
      let response = await nativeFetch(url, options);
      if (enabled && response.status === 401) {await login(deployment); response = await nativeFetch(url, options);}
      return response;
    }
  };
})();
