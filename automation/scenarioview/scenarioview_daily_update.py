"""After-close public-report refresh. No private uploads, model inference or training."""
from pathlib import Path
import argparse,datetime as dt,json,csv,io,re,urllib.request,subprocess,hashlib,os,time
KST=dt.timezone(dt.timedelta(hours=9))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def eligible(now,cfg):
 day=now.strftime('%Y-%m-%d');override=cfg.get('session_overrides',{}).get(day,'15:30')
 if now.weekday()>4 or override is None:return False
 close=dt.datetime.combine(now.date(),dt.time.fromisoformat(override),KST)+dt.timedelta(minutes=1)
 return close<=now<=dt.datetime.combine(now.date(),dt.time(21,0),KST)
def git(args,root):
 p=subprocess.run(['git',*args],cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=90)
 if p.returncode:raise RuntimeError('git operation failed; private details omitted')
 return p.stdout.strip()
def run(cfg,force=False,dry=False):
 now=dt.datetime.now(KST);statepath=Path(cfg['state_path']);state=load(statepath) if statepath.exists() else {};result={'checked_at':now.isoformat(),'inference_rerun':False}
 if not force and not eligible(now,cfg):return dict(result,status='outside_postclose_window')
 site=Path(cfg['site_checkout']);template=Path(cfg['template_path']).read_text(encoding='utf-8');macro=[]
 # Only these public identifiers are allowed into network requests.
 for sid in ['KORLOLITOAASTSAM','KORPRMNTO01GYSAM','LRHUTTTTKRM156S']:
  url='https://fred.stlouisfed.org/graph/fredgraph.csv?id='+sid+'&cosd=2022-01-01&coed='+now.strftime('%Y-%m-%d')
  with urllib.request.urlopen(url,timeout=20) as r:raw=r.read().decode('utf-8-sig')
  pairs=[]
  for row in list(csv.reader(io.StringIO(raw)))[1:]:
   try:
    date=dt.date.fromisoformat(row[0]);value=float(row[1]);assert abs(value)<1e8
    if date<=now.date():pairs.append((date.isoformat(),value))
   except (ValueError,IndexError):continue
  assert pairs,'No valid public observations';date,value=pairs[-1];macro.append(dict(series=sid,observation_date=date,value=value))
 # Private source is inspected locally for freshness only; no values are uploaded.
 feed_date=None
 if cfg.get('local_price_path') and Path(cfg['local_price_path']).exists():
  import pandas as pd
  f=pd.read_parquet(cfg['local_price_path'],columns=['dt','market']);v=f.loc[f.market=='KOSPI','dt'];feed_date=pd.to_datetime(v.astype(str),errors='coerce').max().strftime('%Y-%m-%d') if len(v) else None
 data=load(cfg['evaluation_path']);digest=hashlib.sha256(json.dumps([macro,data],sort_keys=True).encode()).hexdigest()
 result.update(local_data_as_of=feed_date,local_source_current=feed_date==now.strftime('%Y-%m-%d'),fresh_market_signal=False)
 if not dry and state.get('public_digest')==digest:
  save(cfg['last_check_path'],dict(result,status='unchanged'));return dict(result,status='unchanged')
 for m in macro:
  # Anchor each card on its immutable public source ID, not private-report text.
  matches=list(re.finditer(r'<div class="kpi">.*?</div>',template,re.S))
  target=next(x.group(0) for x in matches if '/series/'+m['series']+'"' in x.group(0))
  updated=re.sub(r'<strong>.*?</strong>',f'<strong>{m["value"]:.2f}</strong>',target)
  updated=re.sub(r'관측 \d{4}-\d{2}',f'관측 {m["observation_date"][:7]}',updated)
  template=template.replace(target,updated,1)
 stamp=now.strftime('%Y.%m.%d');template=re.sub(r'발행 \d{4}\.\d{2}\.\d{2}',f'발행 {stamp}',template);template=re.sub(r'<title>ScenarioView \| .*?</title>',f'<title>ScenarioView | {stamp} MiDM × B3</title>',template)
 periods=' / '.join(sorted(set(x['observation_date'][:7] for x in macro)))
 template=template.replace('공개 거시통계 2026년 7~8월',f'공개 거시통계 {periods}')
 template=template.replace('자동 일일 갱신은 미설정','평일 장 마감 후 자동 확인 · 신규 공개 통계가 있을 때 갱신 · 당일 모델 추론 미연결')
 template=template.replace('</main>','<section><h2>자동 갱신 상태</h2><p>공개 통계 확인 '+now.strftime('%Y-%m-%d %H:%M KST')+'</p><p>당일 모델 추론은 연결되지 않았습니다. 모델 비교는 고정 후향 평가이며, 거시통계의 관측일은 각 카드에 표시합니다.</p></section></main>')
 for text in [template,json.dumps(data)]:
  assert not re.search(r'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|LLM_[A-Za-z0-9_-]{15,})\b',text)
  assert not re.search(r'[A-Z]:[\\/]',text) and 'private_local_' not in text
 if dry:return dict(result,status='dry_run_pass',public_digest=digest,public_series=len(macro))
 assert not git(['status','--porcelain'],site),'Site checkout has uncommitted changes'
 git(['pull','--ff-only','origin','gh-pages'],site)
 (site/'index.html').write_text(template,encoding='utf-8');save(site/'macro_latest.json',dict(checked_at=now.isoformat(),series=macro,source='OECD via FRED; revised, not historical-vintage aligned'))
 git(['add','--','index.html','macro_latest.json'],site)
 if git(['diff','--cached','--name-only'],site):
  git(['commit','-m','Refresh public macro report '+now.strftime('%Y-%m-%d')+'; no new model inference'],site);git(['push','origin','gh-pages'],site)
 save(statepath,dict(public_digest=digest,published_at=now.isoformat(),commit=git(['rev-parse','HEAD'],site)));save(cfg['last_check_path'],dict(result,status='published'))
 return dict(result,status='published')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--force-check',action='store_true');a=ap.parse_args();cfg=load(a.config);lock=Path(cfg['state_path']).with_suffix('.lock');lock.parent.mkdir(parents=True,exist_ok=True)
 try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:
  print(json.dumps(dict(status='already_running_or_stale_lock',manual_recovery_required=True)));return
 try:
  os.write(fd,str(os.getpid()).encode());os.close(fd)
  # The scheduler polls every minute; public monthly sources are throttled to 15 minutes.
  last=Path(cfg['last_check_path'])
  if not a.force_check and last.exists() and time.time()-last.stat().st_mtime<900:print(json.dumps(dict(status='public_source_throttled')));return
  print(json.dumps(run(cfg,a.force_check,a.dry_run),ensure_ascii=False))
 except Exception as e:
  save(cfg['last_check_path'],dict(status='failed',error_type=type(e).__name__,checked_at=dt.datetime.now(KST).isoformat()));raise SystemExit(1)
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
