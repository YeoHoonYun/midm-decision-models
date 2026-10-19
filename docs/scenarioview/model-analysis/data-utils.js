export const PERIODS=[1,5,10,20,60,120];
export const finite=n=>typeof n==='number'&&Number.isFinite(n);
export function ranked(rows){let last=null,rank=0;return rows.filter(r=>finite(r.value)).slice().sort((a,b)=>b.value-a.value||a.name.localeCompare(b.name,'ko',{numeric:true})).map((r,i)=>{if(!i||r.value!==last)rank=i+1;last=r.value;return {...r,rank};});}
export function navRows(models,{period,date,industry=false,sector='all'}={}){
  return ranked(models.filter(m=>Boolean(m.sector)===industry&&(sector==='all'||m.sector===sector)).flatMap(m=>{const p=m.periodHistory.find(p=>p.date===date)?.returns[period];return p&&finite(p.value)?[{id:m.id,name:m.name,sector:m.sector,value:p.value,from:p.from,to:date,model:m}]:[];}));
}
export function metricRows(metrics,{period,date,group,market,split,minEvents=10}={}){
  return ranked(metrics.filter(m=>m.horizon===period&&m.reportDate===date&&m.group===group&&m.market===market&&m.split===split&&m.events>=Math.max(1,Number(minEvents)||1)).map(m=>({id:[m.model,m.horizon,m.market,m.split,m.group].join('|'),name:m.model,value:m.netReturnPct,from:m.evaluationFrom,to:m.evaluationTo,metric:m})));
}
export function activityRows(activity,{date,period,market,search=''}={}){return activity.filter(r=>r.date===date&&r.horizon===period&&r.market===market&&r.model.toLowerCase().includes(search.toLowerCase())).sort((a,b)=>b.recommendations-a.recommendations||a.model.localeCompare(b.model));}
