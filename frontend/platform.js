/* Course platform. Values come from read-only evidence and persisted job APIs. */
(() => {
  const base = '/api/v1/platform';
  const esc = escapeHTML;
  const pct = v => v == null ? '未测试' : `${(Number(v) * 100).toFixed(2)}%`;
  const number = v => v == null ? '未测试' : typeof v === 'number' ? v.toFixed(3) : typeof v === 'object' ? JSON.stringify(v) : String(v);
  const json = v => `<pre class="json">${esc(JSON.stringify(v, null, 2))}</pre>`;
  const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  const cell = (...values) => `<tr>${values.map(v => `<td>${v}</td>`).join('')}</tr>`;
  const state = value => `<span class="badge ${value === '已达标' || value === '已实现' ? 'success' : value === '未达标' ? 'danger' : 'warning'}">${esc(value)}</span>`;
  const fileLinks = (items, prefix) => `<div class="artifact-links">${items.map(a => `<a href="${prefix}/${encodeURIComponent(a.name)}">${esc(a.name)} · ${formatBytes(a.bytes)}</a>`).join('')}</div>`;
  let datasets = [], histories = [], currentJob = '', lastSeq = 0, logLines = [], jobBusy = false;
  let lastJob = null, retrievalJob = '', previewValid = false;
  const csvPreview = (items, prefix) => {const files=items.filter(a=>['feature_failures.csv','failed_queries_plaintext.csv','query_results.csv','model_comparison.csv'].includes(a.name));return files.length?`<h3>诊断预览</h3><div class="artifact-links">${files.map(f=>`<button class="button secondary preview-csv" data-url="${prefix}/${encodeURIComponent(f.name)}">${esc(f.name)}</button>`).join('')}</div><div class="csv-preview"></div>`:'';};
  const kindLabel = {formal_test: '正式 test · 只读', selection: '完整 selection · 非test', historical_full: '历史 selection + test', exploratory: '探索性 · 非正式指标', failed_strategy: '负收益方案 · 留档'};
  const statusLabels = {queued: '排队', running: '运行中', completed: '完整完成', failed: '失败 / incomplete', cancelled: '已取消 / incomplete'};

  function metricsPanel(data) {
    const m = data.metrics || {}, a = data.analysis || {};
    let rows = [];
    const push = (label, r, tf = {}) => rows.push(cell(esc(label), pct(r?.top1_accuracy?.value), pct(r?.recall_at_5?.value), pct(tf.tar ?? tf.selection_tar), esc(number(tf.far ?? tf.selection_far ?? tf.empirical_far_all_impostor_trials)), esc(number(tf.threshold))));
    for (const mode of ['plaintext', 'encrypted_restore']) {
      if (m[mode]?.retrieval) push(mode, m[mode].retrieval, m[mode].tar_far_test || m.threshold_selection || m.tar_far_selection || {});
    }
    if (m.retrieval) push('plaintext selection', m.retrieval, m.tar_far_selection);
    for (const [name, model] of Object.entries(m.models || {})) push(`${name} · 探索性`, model.retrieval);
    if (a.baseline_test_retrieval) push('SFace 同一 test 子集对照', a.baseline_test_retrieval, a.baseline_test_tar_far);
    const lead = rows.length ? table(['范围 / 模式', 'Top-1', 'Recall@5', 'TAR', '实际 FAR', '阈值'], rows) : '<div class="notice">此记录未采用统一指标结构，请查看原始 metrics / 报告；缺失数据不按零展示。</div>';
    return lead + `<p class="muted">TAR/FAR 必须结合分母、缺失分数和阈值来源解释。完整历史 SFace 汇总检索包含两个 split，不能与单一 test 直接比较；同 test 对照见 ArcFace 正式 test 分析。</p>` +
      (a.comparison ? table(['同 test 子集指标', 'SFace', 'ArcFace', '差值（百分点）'], a.comparison.map(r => cell(esc(r.metric), pct(r.sface_test), pct(r.arcface_test), esc(number(r.absolute_delta_percentage_points))))) : '') +
      `<details><summary>完整指标、有效/缺失 pair、耗时与阈值来源</summary>${json(m)}</details>` +
      (data.feature_counts ? `<details><summary>媒体有效/无效与特征失败统计</summary>${json(data.feature_counts)}</details>` : '') +
      (a.feature_availability ? `<details><summary>Query / Gallery、恢复一致性、失败原因与时间口径</summary>${json({feature_availability: a.feature_availability, pair_statistics: a.pair_statistics, failure_reasons: a.failure_reasons, plaintext_encrypted_metrics_equal: a.plaintext_encrypted_metrics_equal, plaintext_encrypted_pair_scores_exactly_equal: a.plaintext_encrypted_pair_scores_exactly_equal, restored_sha256_pass_count: a.restored_sha256_pass_count, execution_timing: a.execution_timing})}</details>` : '') +
      `<details><summary>冻结配置 / 数据划分 / 模型校验值</summary>${json(data.config || data.request || {})}</details>`;
  }
  async function loadOverview() {
    const d = await api(`${base}/overview`), m = d.formal_test.metrics?.plaintext || {};
    $('searchModelNote').textContent=d.default_model==='arcface' ? `运行时模型 ArcFace（512维）；阈值 ${d.arcface_threshold} 来自独立 selection。相似度不是身份确认概率。` : `运行时模型 ${d.default_model}；使用该模型预处理，不沿用 ArcFace 阈值。相似度不是身份确认概率。`;
    $('platformKeyDetails').innerHTML=`<h3 class="spaced">运行时后端与密钥管理</h3>${table(['字段','实际配置'],Object.entries({密码后端:d.crypto_backend,库版本:d.crypto_library_version,SM4子实现:d.sm4_implementation,加密特征缓存:d.protected_feature_cache?"已启用":"未启用",密钥版本:d.key_management.key_version,派生:d.key_management.derivation,配置可用:d.key_management.configured?'是':'否',存储:d.key_management.storage,轮换:d.key_management.rotation,授权:d.key_management.authorization}).map(([k,v])=>cell(esc(k),esc(v ?? '不适用'))))}`;
    $('platformOverview').innerHTML = `<p class="muted">运行时默认模型：<strong>${esc(d.default_model)}</strong> · 密码后端：${esc(d.crypto_backend)} / SM4 ${esc(d.sm4_implementation)} · ArcFace 阈值：${esc(d.arcface_threshold)}（独立 selection 冻结）</p><div class="platform-metrics">${[
      ['正式 test Top-1', pct(m.retrieval?.top1_accuracy?.value), '目标 ≥95%'], ['正式 test TAR', pct(m.tar_far_test?.tar), '目标 ≥90% @ FAR=1e-3'], ['正式 test Query', m.retrieval?.query_count ?? '未测试', '不是万人规模验收'], ['已登记数据集', d.datasets.length, '仅白名单路径']
    ].map(([l,v,f]) => `<div><div class="metric-label">${esc(l)}</div><div class="metric-value">${esc(v)}</div><div class="metric-foot">${esc(f)}</div></div>`).join('')}</div><p class="muted spaced">${esc(d.retrieval_note)}</p><details><summary>数据集规模与最近批量任务</summary>${json({datasets:d.datasets, recent_jobs:d.recent_jobs.map(j=>({id:j.id,status:j.status,stage:j.stage}))})}</details>`;
  }
  async function loadHistories() {
    histories = (await api(`${base}/histories`)).items;
    $('historyList').innerHTML = histories.map(h => `<article class="card history-row"><div><h3>${esc(h.name)}</h3><span class="badge">${esc(kindLabel[h.kind] || h.kind)}</span><p class="source spaced">${esc(h.path)}</p></div><button class="button secondary history-open" data-id="${esc(h.id)}">查看指标与报告</button></article>`).join('');
    $('historyList').querySelectorAll('.history-open').forEach(b => b.onclick = () => {
      const h = histories.find(x=>x.id===b.dataset.id);
      $('historyDetail').innerHTML = `<h2>${esc(h.name)}</h2><p class="muted">${esc(kindLabel[h.kind])}；只读历史数据。${h.complete ? '已有完整结果文件' : '完成状态见原始记录，不据此声称正式完成'}</p>${metricsPanel(h)}<h3>结果文件与失败样本</h3>${csvPreview(h.artifacts, `${base}/histories/${encodeURIComponent(h.id)}/preview`)}${fileLinks(h.artifacts, `${base}/histories/${encodeURIComponent(h.id)}/files`)}`;
      $('historyDetail').scrollIntoView({behavior:'smooth',block:'start'});
    });
  }
  async function loadRequirements() {
    const d = await api(`${base}/requirements`);
    $('researchTable').innerHTML = table(['研究内容', '实现状态', '实际能力'],d.research.map(r=>cell(esc(r.name),state(r.status),esc(r.description))));
    $('requirementTable').innerHTML = table(['指标', '目标', '当前实测 / 证据摘要', '达成状态', '数据来源与口径'],d.indicators.map(r=>cell(esc(r.requirement),esc(r.target),esc(number(r.actual)),state(r.status),`<div class="source">${esc(r.source)}</div><p class="muted">${esc(r.note)}</p>`)));
  }
  async function loadBenchmarks() {
    const d = await api(`${base}/benchmarks`);
    $('benchmarkList').innerHTML = `<div class="notice">${esc(d.note)} 当前不从网页重跑性能测试，保留已冻结国密实现。</div>` + d.items.slice().reverse().map(b => {
      const data = b.data, rows=[];
      for(const [algorithm, body] of Object.entries({SM2:data.sm2 || {},SM4:data.sm4 || {}})) {
        for(const [k,v] of Object.entries(body)) {
          if(v && typeof v==='object' && v.median != null) rows.push(cell(esc(`${algorithm} · ${k}`),esc(number(v.median)),esc(number(v.min)),esc(number(v.max)),esc(number(v.stddev)),esc(v.unit)));
        }
      }
      return `<section class="card"><h3>${esc(b.id)}</h3><p class="source">${esc(b.source)}</p>${rows.length ? table(['测量层次 / 操作', '中位数', '最小值', '最大值', '标准差', '单位'],rows) : '<p class="muted">旧版或矩阵记录，查看原始统计口径</p>'}<details><summary>测试环境、轮数与预热</summary>${json(data.environment || {})}</details><details><summary>1080p、分辨率、10路API与原始数据</summary>${json(data)}</details><a class="button secondary" href="${base}/benchmarks/${encodeURIComponent(b.id)}">下载原始 JSON</a></section>`;
    }).join('');
  }
  function batchPayload() {return {dataset_id:$('batchDataset').value, model:$('batchModel').value,kind:'batch',top_k:5};}
  const post = (path, body) => api(`${base}${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  function updateDataset() {
    previewValid=false;$('batchStart').disabled=true;
    const d=datasets.find(x=>x.id===$('batchDataset').value);
    $('datasetDetails').innerHTML=d ? table(['字段','真实登记值'],[['名称',d.name],['身份数',d.identity_count],['Gallery 视频',d.gallery_count],['Query 图像',d.query_count],['帧数',d.frame_count],['帧数口径',d.frame_count_basis],['Manifest',d.manifest],['划分文件',d.identity_split],['评估范围',d.scope],['耗时规划',d.estimated_minutes]].map(([k,v])=>cell(esc(k),esc(v ?? '不可用')))) : '数据不可用';
    $('batchResources').textContent='请点击资源预检';
  }
  async function loadDatasets() {
    datasets=(await api(`${base}/datasets`)).items;
    $('batchDataset').innerHTML=datasets.map(d=>`<option value="${esc(d.id)}" ${d.available?'':'disabled'}>${esc(d.name)}</option>`).join('');
    updateDataset();
    const q=await api(`${base}/datasets/ytf_smoke20/queries`);
    $('datasetQuery').innerHTML=q.items.map(r=>`<option value="${esc(r.query_id)}">${esc(r.query_id)} · ${esc(r.identity_id)} · ${esc(r.split)}</option>`).join('');
  }
  async function loadJobs() {
    const data=await api(`${base}/jobs`);
    $('batchList').innerHTML=data.items.length ? table(['任务 / 类型','模型 / 数据集','状态','进度 / 阶段','操作'],data.items.map(j=>cell(`<code>${esc(j.id.slice(0,12))}</code><p>${j.config.kind==='retrieval'?'实时1:N':'探索性批量'}</p>`,esc(`${j.config.model} / ${j.config.dataset_id}`),`<span class="badge ${j.status==='completed'?'success':j.status==='failed'?'danger':'warning'}">${esc(statusLabels[j.status] || j.status)}</span>`,`${j.processed}/${j.total}<p>${esc(j.stage)}</p>`,`<button class="button secondary job-open" data-id="${esc(j.id)}">查看</button>`))) : '<div class="empty">尚无批量任务；不会自动启动</div>';
    $('batchList').querySelectorAll('.job-open').forEach(b=>b.onclick=()=>selectJob(b.dataset.id));
    const active=data.items.some(j=>['queued','running'].includes(j.status));
    $('batchStart').disabled=active || !previewValid;
    $('datasetSearch').disabled=active;
    return data.items;
  }
  async function selectJob(id) {currentJob=id;lastSeq=0;logLines=[];await updateJob();}
  function renderRetrieval(result) {
    $('datasetResults').innerHTML=`<h3>真实计算结果 · ${esc(result.mode)}</h3><p class="muted">${esc(result.note)} Gallery=${result.gallery_count}，未评分=${result.unscored_pair_count}，总耗时=${Number(result.total_seconds).toFixed(2)} s</p>` + table(['排名','候选视频','身份','相似度','帧号','时间戳'],result.results.map((r,i)=>cell(String(i+1),esc(r.video_id),esc(r.identity_id),Number(r.similarity).toFixed(6),esc(r.candidate_frame_index ?? '未记录'),esc(r.candidate_timestamp_seconds ?? '未记录')))) + `<details><summary>特征可用性和恢复校验</summary>${json(result)}</details>`;
  }
  async function updateJob() {
    if(!currentJob || jobBusy)return;
    jobBusy=true;
    try {
      const j=await api(`${base}/jobs/${encodeURIComponent(currentJob)}`);
      // Drain paginated logs even when the job completed while this browser was closed.
      for(let page=0;page<32;page++) {
        const logs=await api(`${base}/jobs/${encodeURIComponent(currentJob)}/logs?after=${lastSeq}`);
        for(const l of logs.items){lastSeq=l.seq;logLines.push(`${l.created_at} ${l.message}`);}
        logLines=logLines.slice(-500);
        if(logs.items.length<200)break;
      }
      lastJob=j;
      const percent=j.total ? Math.min(100,100*j.processed/j.total):0;
      const result=j.result;
      $('jobDetail').innerHTML=`<div class="toolbar"><div><h2>任务 ${esc(j.id.slice(0,12))}</h2><p class="muted">${esc(statusLabels[j.status])} · ${esc(j.stage)} · 工作计时 ${Number(j.elapsed_seconds).toFixed(1)} s</p></div>${['queued','running'].includes(j.status)?`<button id="cancelJob" class="button secondary" ${j.cancel_requested?'disabled':''}>${j.cancel_requested?'等待安全取消':'请求安全取消'}</button>`:''}</div><progress value="${percent}" max="100"></progress><p class="muted">媒体处理 ${j.processed}/${j.total}。100%媒体处理后仍需评分、完整性核对和报告发布；以任务 completed 状态为准。</p>${j.error?`<div class="status error">${esc(j.error)}</div>`:''}<details><summary>任务配置、状态与时间</summary>${json(j.config)}${json({created_at:j.created_at,started_at:j.started_at,finished_at:j.finished_at,output_dir:j.output_dir,complete:j.complete})}</details><h3>持久化日志（最近500条，可轮询获取全部）</h3><pre class="logs">${esc(logLines.join('\n') || '等待工作进程')}</pre>${result?`<h3>完整结果</h3>${result.result?json(result.result):metricsPanel(result)}${csvPreview(result.artifacts,`${base}/jobs/${encodeURIComponent(j.id)}/preview`)}${fileLinks(result.artifacts,`${base}/jobs/${encodeURIComponent(j.id)}/files`)}`:''}`;
      if($('cancelJob'))$('cancelJob').onclick=async()=>{try{await post(`/jobs/${j.id}/cancel`,{});await updateJob();}catch(e){showStatus('batchStatus',e.message,'error');}};
      if(j.config.kind==='retrieval') {
        retrievalJob=j.id;
        showStatus('retrievalStatus',`任务 ${j.id}：${statusLabels[j.status]} · ${j.stage}`,j.status==='failed'?'error':j.status==='completed'?'success':'');
        if(result?.result)renderRetrieval(result.result);
      }
    } finally {jobBusy=false;}
  }
  $('batchDataset').onchange=updateDataset;
  $('batchModel').onchange=()=>{previewValid=false;$('batchStart').disabled=true;$('batchResources').textContent='模型已改变，请重新预检';};
  $('batchPreview').onclick=async()=>{
    $('batchPreview').disabled=true;
    try{const p=await post('/jobs/preview',batchPayload());previewValid=p.free_disk_bytes>=p.required_disk_bytes;$('batchResources').innerHTML=`<p>剩余磁盘：${formatBytes(p.free_disk_bytes)}；保守最低需求：${formatBytes(p.required_disk_bytes)}</p><p>运行范围：${esc(p.config.evaluation_scope)}；模式：${esc(p.modes.join(' + '))}；预计 ${esc(p.config.estimated_duration)}</p><details><summary>已校验配置与模型 SHA-256</summary>${json(p.config)}</details>`;showStatus('batchStatus',previewValid?'预检通过。点击“启动独立实验”才会运行。':'磁盘不足，未启动',previewValid?'success':'error');await loadJobs();}
    catch(e){previewValid=false;$('batchStart').disabled=true;showStatus('batchStatus',e.message,'error');}finally{$('batchPreview').disabled=false;}
  };
  $('batchStart').onclick=async()=>{
    if(!previewValid)return;
    $('batchStart').disabled=true;
    try{const j=await post('/jobs',batchPayload());showStatus('batchStatus',`任务已入队 ${j.id}`,'success');await selectJob(j.id);await loadJobs();}
    catch(e){showStatus('batchStatus',e.message,'error');await loadJobs();}
  };
  $('datasetSearch').onclick=async()=>{
    const top_k=Number($('retrievalK').value);
    if(!Number.isInteger(top_k)||top_k<1||top_k>100)return showStatus('retrievalStatus','Top-K 必须为1–100整数','error');
    $('datasetSearch').disabled=true;
    try{const j=await post('/jobs',{kind:'retrieval',dataset_id:'ytf_smoke20',model:$('retrievalModel').value,query_id:$('datasetQuery').value,mode:$('retrievalMode').value,top_k});retrievalJob=j.id;await selectJob(j.id);await loadJobs();}
    catch(e){showStatus('retrievalStatus',e.message,'error');$('datasetSearch').disabled=false;}
  };
  $('batchRefresh').onclick=()=>loadJobs().then(updateJob).catch(e=>renderError('batchList',e));
  $('historyRefresh').onclick=()=>loadHistories().catch(e=>renderError('historyList',e));
  window.platformViewChanged=name=>{
    const fn={overview:loadOverview,experiments:loadHistories,batch:loadJobs,requirements:loadRequirements,benchmarks:loadBenchmarks}[name];
    if(fn)fn().catch(e=>renderError({overview:'platformOverview',experiments:'historyList',batch:'batchList',requirements:'requirementTable',benchmarks:'benchmarkList'}[name],e));
  };
  document.addEventListener('click',async e=>{
    const preview=e.target.closest('.preview-csv');
    if(preview){
      const output=preview.closest('.card').querySelector('.csv-preview');
      try{const d=await api(preview.dataset.url);output.innerHTML=`<p class="muted">${esc(d.note)} ${d.truncated?'（有更多记录）':''}</p>${d.items.length?table(d.columns,d.items.map(r=>cell(...d.columns.map(c=>esc(r[c]))))):'<div class="empty">文件中无诊断记录</div>'}`;}catch(error){output.textContent=error.message;}
      return;
    }
    const b=e.target.closest('.verify-video,.restore-video');if(!b)return;
    const id=b.dataset.taskId, output=b.closest('.details').querySelector('.verify-result');b.disabled=true;output.textContent='正在逐块认证、恢复并验证 SHA-256...';
    try{
      if(b.classList.contains('verify-video')){const r=await post(`/videos/${encodeURIComponent(id)}/verify`,{});output.textContent=`认证与恢复校验通过；SHA-256=${r.sha256}；${r.bytes} bytes；${r.restore_seconds.toFixed(3)} s。${r.note}`;}
      else{const response=await window.facecheckAuth.fetch(`${base}/videos/${encodeURIComponent(id)}/restore`,{method:'POST'});if(!response.ok)throw new Error((await response.json()).detail);const blob=await response.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='restored.mp4';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);output.textContent='块认证与恢复校验通过，已下载恢复视频。';}
      await refreshAudit();
    }catch(error){output.textContent=`失败：${error.message}`;}finally{b.disabled=false;}
  });
  const auditFilter=$('auditFilter').onclick;
  $('auditFilter').onclick=()=>{auditFilter();const suffix=currentAuditFilter?`&task_id=${encodeURIComponent(currentAuditFilter)}`:'';$('auditCsv').href=`${base}/audit/export?format=csv${suffix}`;$('auditReport').href=`${base}/audit/export?format=md${suffix}`;};
  Promise.allSettled([loadOverview(),loadDatasets(),loadJobs()]).then(results=>{
    results.forEach((r,i)=>{if(r.status==='rejected')renderError(['platformOverview','datasetDetails','batchList'][i],r.reason);});
  });
  setInterval(async()=>{
    try{const jobs=await loadJobs();if(currentJob && (!lastJob||['queued','running'].includes(lastJob.status)))await updateJob();else if(!currentJob){const active=jobs.find(j=>['queued','running'].includes(j.status));if(active)await selectJob(active.id);}}
    catch(e){showStatus('batchStatus',`刷新失败：${e.message}`,'error');}
  },4000);
})();
