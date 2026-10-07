param([switch]$DryRun,[switch]$ForceCheck)
$ErrorActionPreference='Stop'
$config=Join-Path $PSScriptRoot 'scenarioview_daily.local.json'
$settings=Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
$logRoot=Join-Path (Split-Path $settings.state_path -Parent) 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$argsList=@('-I',(Join-Path $PSScriptRoot 'scenarioview_daily_update.py'),'--config',$config)
if($DryRun){$argsList+='--dry-run'}
if($ForceCheck){$argsList+='--force-check'}
$log=Join-Path $logRoot ((Get-Date -Format 'yyyyMMddTHHmmss')+'.log')
& $settings.python_exe @argsList *> $log
exit $LASTEXITCODE
