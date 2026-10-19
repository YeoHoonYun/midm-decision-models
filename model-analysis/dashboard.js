import { PERIODS,finite,navRows,metricRows,activityRows } from './data-utils.js';
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const date=v=>{const s=String(v||'').replaceAll('-','');return /^\d{8}$/.test(s)?`${s.slice(0,4)}.${s.slice(4,6)}.${s.slice(6)}`:'미제공';};
const pct=v=>finite(v)?`${v>0?'+':''}${v.toFixed(2)}%`:'—',pp=v=>finite(v)?`${v>0?'+':''}${v.toFixed(2)}%p`:'—',dir=v=>v>0?'positive':v<0?'negative':'',count=v=>v===null||v===undefined?'—':Number(v).toLocaleString('ko-KR');
const groups={daily:'일일 학습 모델',legacy:'기존 모델 평가',recommendations:'추천 후 확정 추적'},splits={validation:'검증',test:'테스트 · 진단용',legacy_test:'과거 테스트',forward_realized:'추천 후 확정 표본',unknown:'구간 미확인'};
const state={data:null,tab:'scenarios',market:'KR',scenarioMarket:'all',basis:'nav',period:20,date:null,industryDate:null,report:null,activityDate:null,sector:'all',group:'recommendations',split:'forward_realized',minEvents:10,limit:10,rankModel:'',industryModel:'',activityModel:'',page:0,signalsOpen:false,signalKind:'all',signalPage:0};
let rankedRows=[],loading=false;
const select=(id,label,options,value)=>`<label>${label}<select id="${id}" aria-label="${label}">${options.map(([v,t])=>`<option value="${esc(v)}" ${String(value)===String(v)?'selected':''}>${esc(t)}</option>`).join('')}</select></label>`;
function modelPicker(id,entries,key){
  const options=[...new Map(entries).entries()].sort((a,b)=>a[1].localeCompare(b[1],'ko',{numeric:true}));
  if(!options.some(([value])=>value===state[key]))state[key]='';
  return select(id,'모델 선택',[['',`전체 모델 (${options.length}개)`],...options],state[key]).replace('<label>','<label class="model-picker">');
}
const table=(headers,body,note='')=>`<section class="panel table-panel"><div class="table-scroll"><table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${body||`<tr><td colspan="${headers.length}"><div class="empty">조건에 맞는 기록이 없습니다. 기간·기준일·필터를 변경해 보세요.</div></td></tr>`}</tbody></table></div><p class="table-caption">${esc(note)}</p></section>`;
const periodButtons=()=>`<div class="periods" aria-label="분석 기간">${PERIODS.map(n=>`<button data-period="${n}" class="${state.period===n?'active':''}" aria-pressed="${state.period===n}">${n}일</button>`).join('')}</div>`;
function scenarios(){
  const data=state.data.scenario,assets=data.assets.filter(a=>state.scenarioMarket==='all'||a.market===state.scenarioMarket);
  return `<section class="panel"><div class="section-head"><div><p class="eyebrow">SCENARIO LIBRARY</p><h2>지수·주요 기업 시나리오</h2><p>원본 보고서의 자료 기준일과 적용 모델을 확인하세요.</p></div><span class="pill">시나리오 보고서 발행 ${date(data.issueDate)}</span></div><div class="filters">${select('scenario-market','대상 시장',[['all','전체 시장'],['KR','KR 한국'],['US','US 미국']],state.scenarioMarket)}</div></section><div class="scenario-grid">${assets.map(a=>`<article class="scenario-card ${a.market.toLowerCase()}"><span class="market">${a.market==='KR'?'KOREA MARKET':'US MARKET'} · ${a.kind==='index'?'INDEX':'STOCK'}</span><h3>${esc(a.name)}</h3><div class="asset-code">${esc(a.id)}</div><dl><div><dt>자료 기준일</dt><dd>${date(a.dataDate)}</dd></div><div><dt>선택 모델</dt><dd>로컬 MiDM</dd></div><div><dt>B3 적용</dt><dd>${a.b3Applied?'지수 전용 적용':'개별 종목 미적용'}</dd></div></dl><a class="report-link" href="${esc(a.report)}" target="_blank" rel="noreferrer">시나리오·대응 조건 보기 ↗</a></article>`).join('')}</div><p class="source-note">자료 기준일 이후의 시장 변화는 원본 보고서에 반영되지 않을 수 있습니다. 개별 기업의 전이 성능은 별도 검증 전이며, 목록은 현재 전체시장 시가총액 순위 인증이 아닙니다.</p>`;
}
function rankings(industry=false){
  const d=state.data,basis=industry?'nav':state.basis;let filters='',rankDate,excluded=0,modelOptions=[];const modelKey=industry?'industryModel':'rankModel';
  if(basis==='nav'){
    const models=d.models.filter(m=>Boolean(m.sector)===industry&&(state.sector==='all'||!industry||m.sector===state.sector)),dates=new Map();
    for(const m of models)for(const p of m.periodHistory)dates.set(p.date,(dates.get(p.date)||0)+1);
    modelOptions=models.map(m=>[m.id,m.name]);
    const options=[...dates].sort(([a],[b])=>b.localeCompare(a)),key=industry?'industryDate':'date';
    if(!dates.has(state[key]))state[key]=options.find(([,n])=>n>=2)?.[0]||options[0]?.[0]||null;rankDate=state[key];
    filters=select('rank-date','비교 종료일',options.map(([day,n])=>[day,`${date(day)} · ${n}개 기록`]),rankDate);
    if(industry)filters+=select('sector','산업',[['all','전체 산업'],...[...new Set(d.models.map(m=>m.sector).filter(Boolean))].sort().map(s=>[s,s])],state.sector);
    rankedRows=navRows(d.models,{period:state.period,date:rankDate,industry,sector:industry?state.sector:'all'});excluded=models.length-rankedRows.length;
  }else{
    const available=d.metrics.filter(m=>m.market===state.market&&m.group===state.group),availableSplits=[...new Set(available.map(m=>m.split))];
    if(!availableSplits.includes(state.split))state.split=availableSplits.includes('validation')?'validation':availableSplits[0]||'unknown';
    modelOptions=available.filter(m=>m.split===state.split).map(m=>[m.model,m.model]);
    const dates=[...new Set(available.filter(m=>m.split===state.split&&m.horizon===state.period).map(m=>m.reportDate))].sort().reverse();
    if(!dates.includes(state.report))state.report=dates[0]||null;rankDate=state.report;
    filters=select('group','자료 구분',Object.entries(groups),state.group)+select('market','시장',[['KR','KR 한국'],['US','US 미국']],state.market)+select('split','평가 구간',availableSplits.map(s=>[s,splits[s]||s]),state.split)+select('rank-date','보고서 기준일',dates.map(day=>[day,date(day)]),rankDate)+select('samples','최소 확정 표본',[[1,'1건 이상'],[10,'10건 이상'],[30,'30건 이상'],[100,'100건 이상']],state.minEvents);
    rankedRows=metricRows(d.metrics,{period:state.period,date:rankDate,group:state.group,market:state.market,split:state.split,minEvents:state.minEvents});excluded=available.filter(m=>m.split===state.split&&m.horizon===state.period&&m.reportDate===rankDate).length-rankedRows.length;
  }
  const populationCount=rankedRows.length;filters+=modelPicker('rank-model',modelOptions,modelKey);
  if(state[modelKey])rankedRows=rankedRows.filter(r=>(basis==='nav'?r.id:r.name)===state[modelKey]);
  const visible=state.limit==='all'?rankedRows:rankedRows.slice(0,Number(state.limit)),leader=rankedRows[0];
  return `<section class="panel"><div class="section-head"><div><p class="eyebrow">${industry?'SECTOR PERFORMANCE':'TOP RETURN MODELS'}</p><h2>${industry?'산업별 모델 수익률':'기간별 수익률 상위 모델'}</h2><p>${basis==='nav'?'페이퍼 NAV 변화율 · 최근 N개 일별 관측 구간':'목표 보유기간을 채운 확정 표본의 평균 순수익률'}</p></div><span class="pill">${basis==='nav'?'KR 페이퍼 시뮬레이션':'비용 가정 30bp 차감'}</span></div>${periodButtons()}<div class="filters">${industry?'':select('basis','수익률 기준',[['nav','페이퍼 자산 변화율'],['horizon','보유기간별 평균 순수익률']],state.basis)}${filters}${select('limit','표시 개수',[[5,'상위 5개'],[10,'상위 10개'],[20,'상위 20개'],['all','전체']],state.limit)}</div><p class="source-note">${basis==='nav'?'N일은 원본 일별 NAV의 N개 관측 구간입니다. 같은 종료일과 N+1개 기록이 있는 모델만 비교합니다. 기본 종료일은 2개 이상을 비교할 수 있는 가장 최근 날짜입니다.':'같은 자료·시장·평가 구간·보고서·보유기간끼리 비교합니다. 모델별 실제 평가 기간과 표본은 다를 수 있습니다. 검증과 테스트를 섞지 않습니다.'}${industry?' 산업은 원본 모델명에 명시된 분류입니다. 이름의 실전·관찰·탈락 표시는 원본 분류명이며 여기서는 모두 페이퍼 자산입니다.':''}</p>${leader?`<div class="leader"><div><p class="eyebrow">${state[modelKey]?'SELECTED MODEL':'PERIOD LEADER'} · ${state.period}${basis==='nav'?'관측':'거래일 보유'}</p><h3>${esc(leader.name)}</h3><p>${date(rankDate)} 기준 · ${rankedRows.length}개 순위 산출${leader.sector?' · '+esc(leader.sector):''}</p></div><strong class="${dir(leader.value)}">${pct(leader.value)}</strong></div>`:''}</section><div class="table-heading"><h2>${state.period}일 수익률 순위</h2><span>전체 ${populationCount}개 산출 · ${excluded}개 제외${state[modelKey]?` · 선택 모델 ${rankedRows.length}개`:''}</span></div>${table(['순위','모델 / 구분',basis==='nav'?'자산 변화율':'평균 순수익률',basis==='nav'?'비교 구간':'평가 기간 / 표본','직전 보고서·누적 대비','보기'],visible.map((r,i)=>`<tr><td><span class="rank">${r.rank}</span></td><td><strong class="model-name">${esc(r.name)}</strong><small>${esc(r.sector||r.metric?.market||'KR 페이퍼')}</small></td><td><strong class="return ${dir(r.value)}">${pct(r.value)}</strong></td><td>${r.from?`${date(r.from)} → ${date(r.to)}`:'평가 구간 미제공'}<small>${r.metric?`${count(r.metric.events)}건 · 신호 ${date(r.metric.latestSignal||r.metric.maturedThrough)}`:`${state.period+1}개 일별 관측`}</small></td><td>${pp(r.metric?r.metric.deltaPp:r.model.to===rankDate?r.model.deltaPp:null)}${r.model?'<small>누적 수익률의 직전 관측 대비</small>':''}</td><td><button data-detail="${i}">성과 상세</button></td></tr>`).join(''),'모델 선택 후에도 전체 비교 대상에서의 순위를 유지합니다. 동률은 공동 순위 · 미산출은 제외 · 이 순위는 운영 모델이나 주문을 변경하지 않습니다.')}`;
}
function activityReturn(row){
  const metric=state.data.metrics.filter(m=>m.group==='recommendations'&&m.market===row.market&&m.horizon===row.horizon&&m.model===row.model).sort((a,b)=>b.reportDate.localeCompare(a.reportDate))[0];
  return metric?`<strong class="${dir(metric.netReturnPct)}">${pct(metric.netReturnPct)}</strong><small>${count(metric.events)}건 · ${date(metric.reportDate)} 보고서</small>`:'—<small>동일 모델 집계 미제공</small>';
}
function activity(){
  const d=state.data,available=d.activity.filter(r=>r.market===state.market&&r.horizon===state.period),dates=[...new Set(available.map(r=>r.date))].sort().reverse();
  if(!dates.includes(state.activityDate))state.activityDate=dates[0]||null;
  const picker=modelPicker('activity-model',available.map(r=>[r.model,r.model]),'activityModel');
  const rows=activityRows(d.activity,{date:state.activityDate,period:state.period,market:state.market}).filter(r=>!state.activityModel||r.model===state.activityModel),sum=k=>rows.reduce((n,r)=>n+r[k],0),meta=d.marketStatus.find(m=>m.market===state.market),maxPage=Math.max(0,Math.ceil(rows.length/20)-1);state.page=Math.min(state.page,maxPage);
  return `<section class="panel"><div class="section-head"><div><p class="eyebrow">MODEL SIGNAL / PAPER ACTIVITY</p><h2>추천과 가상 매수·매도 경과</h2><p>추적 원장을 모델·일자별로 집계했습니다. 실제 계좌 주문·체결 내역은 아닙니다.</p></div><span class="pill">${state.market} 메일 산출 ${date(meta?.edition)}</span></div>${periodButtons()}<div class="filters">${select('market','시장',[['KR','KR 한국'],['US','US 미국']],state.market)}${select('activity-date','집계일',dates.map(v=>[v,date(v)]),state.activityDate)}${picker}</div><div class="activity-summary">${[['추천 신호',sum('recommendations')],['가상 매수·진입',sum('buys')],['가상 매도·종료',sum('sells')]].map(([label,n])=>`<div><span>${label}</span><strong>${meta?.available?count(n):'—'}</strong></div>`).join('')}</div><p class="source-note">${esc(d.policy.activity)} 기업행동 의심 표본은 진입·종료 집계에서 제외합니다. 추천 수는 선택한 신호일, 가상 매수·매도는 각각의 진입일·실현 종료일 기준입니다.</p><p class="source-note">최신 추적 신호 ${date(meta?.latestSignal)} · ${state.period}거래일 보유 가정 · ${date(state.activityDate)} 집계 · 새 발행일이 새 추론이나 오늘의 신호를 뜻하지 않습니다.</p></section><div class="table-heading"><h2>모델별 ${date(state.activityDate)} 활동</h2><span>${rows.length}개 모델 · ${state.page+1}/${maxPage+1} 페이지</span><div><button id="prev" ${state.page===0?'disabled':''}>이전</button> <button id="next" ${state.page===maxPage?'disabled':''}>다음</button></div></div>${table(['모델','집계일','추천 신호','가상 매수·진입','가상 매도·종료','확정 평균 순수익률','기업행동 의심 제외'],rows.slice(state.page*20,(state.page+1)*20).map(r=>`<tr><td><strong class="model-name">${esc(r.model)}</strong><small>${r.market} · ${r.horizon}거래일 가정</small></td><td>${date(r.date)}</td><td>${count(r.recommendations)}</td><td>${count(r.buys)}</td><td>${count(r.sells)}</td><td>${activityReturn(r)}</td><td>${count(r.quarantined)}</td></tr>`).join(''),'수익률은 일일 수익률이 아닌 최신 보고서의 선택 보유기간 확정 표본 평균입니다. 모델별 추천이 겹칠 수 있으므로 시장 전체 고유 종목 수가 아닙니다. 실제 수량·체결가격·개인 잔고는 공개하지 않습니다.')}`;
}
function signalPanel(){
  const kinds={recommendation:'추천 후보',entry:'가상 매수·진입',exit:'가상 매도·기간종료'};
  const rows=(state.data.signalDetails||[]).filter(r=>r.market===state.market&&r.horizon===state.period&&r.date===state.activityDate&&(state.signalKind==='all'||r.kind===state.signalKind)&&(!state.activityModel||r.model===state.activityModel)).sort((a,b)=>a.kind.localeCompare(b.kind)||a.model.localeCompare(b.model)||a.symbol.localeCompare(b.symbol));
  const pages=Math.max(1,Math.ceil(rows.length/25));state.signalPage=Math.min(state.signalPage,pages-1);
  return `<details id="signal-details" class="signal-details" ${state.signalsOpen?'open':''}><summary><span>종목별 추천·가상 매수·매도 상세</span><small>${count(rows.length)}건 · 펼치기 / 접기</small></summary><div class="signal-body"><p class="source-note">${esc(state.data.policy.signalDetails)} 아래 신호에는 실제 계좌 주문·체결, 가격·수량·모델 점수를 포함하지 않습니다. 상단과 같은 시장·집계일·보유기간·모델 선택 조건을 사용합니다.</p><div class="filters">${select('signal-kind','신호 구분',[['all','전체 신호'],...Object.entries(kinds)],state.signalKind)}</div><div class="table-heading"><h2>${date(state.activityDate)} 종목 목록</h2><span>${state.signalPage+1}/${pages} 페이지</span><div><button id="signal-prev" ${state.signalPage===0?'disabled':''}>이전</button> <button id="signal-next" ${state.signalPage===pages-1?'disabled':''}>다음</button></div></div>${table(['구분','종목 / 코드','모델','기준일 / 보유기간'],rows.slice(state.signalPage*25,(state.signalPage+1)*25).map(r=>`<tr><td><span class="pill">${esc(kinds[r.kind]||r.kind)}</span></td><td><strong>${esc(r.name)}</strong><small>${esc(r.symbol)} · ${r.market}</small></td><td><span class="model-name">${esc(r.model)}</span></td><td>${date(r.date)}<small>${r.horizon}거래일 보유 가정</small></td></tr>`).join(''),'시장별 최근 10개 활동일의 상세 목록입니다. 이전 날짜에는 상단 집계만 제공될 수 있습니다. 가상 매도는 추적 규칙상 종료이며 신규 매도 예측으로 해석하지 않습니다.')}</div></details>`;
}
function chart(history){
  const rows=history.filter(r=>finite(r.returnPct));if(rows.length<2)return '';
  const min=Math.min(...rows.map(r=>r.returnPct)),max=Math.max(...rows.map(r=>r.returnPct)),span=max-min||1,w=550,h=160;
  const y=n=>15+(max-n)/span*115,path=rows.map((r,i)=>`${i?'L':'M'}${10+i*(w-90)/(rows.length-1)},${y(r.returnPct)}`).join(' ');
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="최근 공개 누적 수익률 추이"><line x1="10" x2="470" y1="15" y2="15"/><line x1="10" x2="470" y1="130" y2="130"/><path d="${path}"/><text x="485" y="20">${pct(max)}</text><text x="485" y="135">${pct(min)}</text><text x="10" y="155">${date(rows[0].date)}</text><text x="470" y="155" text-anchor="end">${date(rows.at(-1).date)}</text></svg>`;
}
function details(index){
  const r=rankedRows[index];if(!r)return;const m=r.metric,model=r.model;
  const fields=[['모델',r.name],['선택한 순위 기간',`${state.period}${m?'거래일 보유':'개 일별 관측 구간'}`],['순위 기준 수익률',pct(r.value)],['순위 기준일',date(m?.reportDate||r.to)],...(m?[['시장 / 평가',`${m.market} · ${splits[m.split]||m.split}`],['확정 표본',`${count(m.events)}건`],['평가 기간',m.evaluationFrom?`${date(m.evaluationFrom)} → ${date(m.evaluationTo)}`:'원본 미제공'],['성숙 신호일',date(m.maturedThrough)],['최근 추천 신호일',date(m.latestSignal)],['총수익률 / 순수익률',`${pct(m.grossReturnPct)} / ${pct(m.netReturnPct)}`],['이전 보고서',date(m.previousReportDate)],['이전 대비',pp(m.deltaPp)]]:[['모델 최신 기록일',date(model.to)],['모델 누적 수익률 · 최신',pct(model.cumulativeReturnPct)],['직전 누적 대비 · 최신',pp(model.deltaPp)]])];
  $('detail-title').textContent=m?'보유기간별 성과 상세':'페이퍼 모델 자산 수익률';
  $('detail-content').innerHTML=`<dl class="details">${fields.map(([label,value])=>`<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>${model?chart(model.history)+`<p class="source-note">아래는 모델 최신 기록 ${date(model.to)} 기준입니다. 선택한 순위 기준일과 다를 수 있습니다.</p><div class="detail-periods">${PERIODS.map(h=>`<div><small>${h}관측 구간</small><strong class="${dir(model.periods[h]?.returnPct)}">${pct(model.periods[h]?.returnPct)}</strong></div>`).join('')}</div>`:''}<p class="source-note">${esc(state.data.policy.returns)} 개인 계좌의 수익률·주문 내역은 포함하지 않습니다.</p>`;
  $('detail').showModal();
}
function bind(){
  const detailPanel=$('signal-details');if(detailPanel)detailPanel.ontoggle=()=>{state.signalsOpen=detailPanel.open;};
  const kind=$('signal-kind');if(kind)kind.onchange=()=>{state.signalKind=kind.value;state.signalPage=0;state.signalsOpen=true;render();};
  if($('signal-prev'))$('signal-prev').onclick=()=>{state.signalPage--;render();};if($('signal-next'))$('signal-next').onclick=()=>{state.signalPage++;render();};
  document.querySelectorAll('[data-period]').forEach(b=>b.onclick=()=>{state.period=Number(b.dataset.period);state.page=0;render();});
  for(const [id,key]of[['scenario-market','scenarioMarket'],['basis','basis'],['group','group'],['market','market'],['split','split'],['sector','sector'],['limit','limit'],['samples','minEvents']]){const input=$(id);if(input)input.onchange=()=>{state[key]=input.value;if(['basis','group','market','split'].includes(key))state.report=null;if(key==='sector')state.industryDate=null;state.page=0;render();};}
  const day=$('rank-date');if(day)day.onchange=()=>{state[state.tab==='industries'?'industryDate':state.basis==='nav'?'date':'report']=day.value;render();};
  const act=$('activity-date');if(act)act.onchange=()=>{state.activityDate=act.value;state.page=0;render();};
  for(const [id,key]of[['rank-model',state.tab==='industries'?'industryModel':'rankModel'],['activity-model','activityModel']]){const picker=$(id);if(picker)picker.onchange=()=>{state[key]=picker.value;state.page=0;state.signalPage=0;render();};}
  document.querySelectorAll('[data-detail]').forEach(b=>b.onclick=()=>details(Number(b.dataset.detail)));
  if($('prev'))$('prev').onclick=()=>{state.page--;render();};if($('next'))$('next').onclick=()=>{state.page++;render();};
}
function render(){if(!state.data)return;$('content').innerHTML=state.tab==='scenarios'?scenarios():state.tab==='activity'?activity()+signalPanel():rankings(state.tab==='industries');document.querySelectorAll('[data-tab]').forEach(b=>{const active=b.dataset.tab===state.tab;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});bind();}
async function load(){
  if(loading)return;loading=true;$('refresh').disabled=true;
  try{const response=await fetch('data/latest.json',{cache:'no-cache',credentials:'omit',signal:AbortSignal.timeout(20000)});if(!response.ok)throw Error('snapshot unavailable');const d=await response.json();if(d.schemaVersion!==1||!Array.isArray(d.models)||!Array.isArray(d.activity))throw Error('unsupported snapshot');state.data=d;
    const stamp=new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',dateStyle:'medium',timeStyle:'short'}).format(new Date(d.generatedAt));$('published').innerHTML=`공개 데이터 발행 ${date(d.publicationDate)}<br>${esc(stamp)} KST`;
    $('status').className=d.warnings.length?'notice error':'notice';$('status').textContent=d.warnings.length?d.warnings.join(' '):'하루 단위로 발행한 공개 집계입니다. 보고서·모델·신호의 자료 기준일은 각각 표시됩니다.';
    $('summary').innerHTML=[['시나리오 대상',`${d.scenario.assets.length}개`,'한국·미국 지수와 주요 기업'],['페이퍼 모델',`${d.models.length}개`,'일반 모델·산업별 모델 분리'],['보유기간',`${d.periods.length}개`,'1 · 5 · 10 · 20 · 60 · 120'],['공개 범위','집계·종목 신호','개인 계좌·가격·점수 제외']].map(([a,b,c])=>`<div><span>${esc(a)}</span><strong>${esc(b)}</strong><small>${esc(c)}</small></div>`).join('');render();
  }catch{$('status').className='notice error';$('status').textContent=state.data?'최신 파일을 확인하지 못했습니다. 이전 발행 자료를 표시합니다.':'공개 데이터 파일을 불러오지 못했습니다. 잠시 후 최신 발행 확인을 눌러 주세요.';}finally{loading=false;$('refresh').disabled=false;}
}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{state.tab=b.dataset.tab;history.replaceState(null,'',`#${state.tab}`);render();});
if(['scenarios','models','industries','activity'].includes(location.hash.slice(1)))state.tab=location.hash.slice(1);
$('refresh').onclick=load;load();
