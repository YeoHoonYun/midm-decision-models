const $=s=>document.querySelector(s),pct=x=>x==null?'—':(100*x).toFixed(2)+'%',names={frequency:'과거 빈도',price_us:'가격+미국 ET',flow:'가격+미국+수급 ET'},classes={up_only:'상방만',down_only:'하방만',up_then_down:'상방→하방',down_then_up:'하방→상방',neither:'양쪽 미도달'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function table(head,rows){return '<div class="table-scroll"><table><thead><tr>'+head.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
if($('#window'))fetch('report-data.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(data=>{
 function render(){const w=Number($('#window').value),n=Number($('#top-n').value),ms=data.topn.filter(x=>x.window===w&&x.N===n);
 $('#coverage-chart').innerHTML=ms.map(m=>'<div class="bar-card"><div class="bar-title"><b>'+names[m.model]+'</b><span>'+pct(m.coverage)+' · 95% 구간 '+pct(m.ci_block5[0])+'–'+pct(m.ci_block5[1])+'</span></div><div class="bar-track"><div class="bar" style="width:'+m.coverage*100+'%"></div><div class="ci-line" style="left:'+m.ci_block5[0]*100+'%;width:'+(m.ci_block5[1]-m.ci_block5[0])*100+'%"></div></div></div>').join('');
 $('#metric-table').innerHTML=table(['방법','포함률','양방향 도달 누락률','Brier(낮을수록)','전체90%·각유형80%'],ms.map(m=>[names[m.model],pct(m.coverage),pct(m.both_barrier_miss),m.brier.toFixed(3),m.meets_descriptive_threshold?'표본 내 충족'+(n===5?' (정의상 상한)':''):'미충족']));
 $('#class-table').innerHTML=table(['방법',...Object.values(classes)],ms.map(m=>[names[m.model],...m.class_coverage.map(c=>pct(c.coverage)+' / n='+c.n)]));
 const ds=data.paired_ml_differences.filter(d=>d.window===w&&d.N===n);
 const pp=x=>(x>=0?'+':'')+(x*100).toFixed(2);
 $('#paired-table').innerHTML=ds.length?table(['같은N 비교','차이(%p)','5일 블록95%(%p)'],ds.map(d=>[names[d.left]+' − '+names[d.right],pp(d.delta),'['+d.ci_block5.map(pp).join(', ')+']'])):'<p>대응 차이는Top1·2·3에서 계산했습니다.</p>';
 const ss=data.seeds.filter(s=>s.window===w&&s.N===n);
 $('#seed-table').innerHTML=ss.length?table(['방법','5시드 평균','표준편차(%p)','최소–최대','집합 Jaccard'],ss.map(s=>[names[s.model],pct(s.mean),(100*s.sd).toFixed(2),pct(s.min)+'–'+pct(s.max),s.mean_pairwise_set_jaccard.toFixed(3)])):'<p>시드별 요약은Top1·2·3에서 계산했습니다. 후보 수를 바꾸어 보세요.</p>';
 const gs=data.groups.filter(g=>g.window===w&&g.N===n&&g.model==='flow');
 $('#group-table').innerHTML=gs.length?table(['분류','그룹','관측 수','날짜 수','포함률'],gs.map(g=>[g.dimension==='industry'?'현재 업종':g.dimension==='regime'?'거시 환경':'상위10 사후부분집합',g.group,g.rows,g.dates,pct(g.coverage)])):'<p>그룹 요약은Top1·2·3에서 계산했습니다.</p>';
 const cs=data.company_history.filter(c=>c.window===w&&c.N===n);
 $('#company-history').innerHTML=cs.length?table(['기업','실현 범주 포함/관측','과거 포함률'],cs.map(c=>[c.name,c.hits+'/'+c.rows,pct(c.coverage)])):'<p>기업별 요약은Top1·2·3에서 계산했습니다.</p>';
 }
 $('#window').addEventListener('change',render);$('#top-n').addEventListener('change',render);render();
}).catch(err=>{$('#metric-table').textContent='집계 자료를 읽지 못했습니다. 공개 JSON 또는 전체 연구 보고서를 확인하세요.';console.error(err)});
if(location.hash.startsWith('#c')){const item=document.getElementById(location.hash.slice(1));if(item?.tagName==='DETAILS')item.open=true}
document.addEventListener('click',ev=>{const a=ev.target.closest('a[href^="#c"]');if(a){const d=document.querySelector(a.getAttribute('href'));if(d?.tagName==='DETAILS')d.open=true}});
