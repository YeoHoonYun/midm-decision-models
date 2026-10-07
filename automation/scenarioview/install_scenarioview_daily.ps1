$ErrorActionPreference='Stop'
if((Get-TimeZone).Id -ne 'Korea Standard Time'){throw 'Task registration requires Korea Standard Time.'}
$taskName='ScenarioView-KOSPI-AfterClose'
$script=Join-Path $PSScriptRoot 'run_scenarioview_daily.ps1'
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$script+'"') -WorkingDirectory $PSScriptRoot
$trigger=New-ScheduledTaskTrigger -Daily -At '15:31'
$trigger.Repetition=(New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Hours 5 -Minutes 29)).Repetition
$settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 8) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'After-close public macro/report refresh; no private uploads; no daily model inference.' -Force | Select-Object TaskName,State
