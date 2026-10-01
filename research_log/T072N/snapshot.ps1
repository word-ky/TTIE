$ErrorActionPreference='Stop'
$env:AUTODL_CONFIG_PATH='D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\config.json'
. 'D:\work\claude-autodl\autodl-workflow-clean\scripts\Autodl.Common.ps1'
$out='D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\T072N-work\research_log\T072N\snapshot.json'
if(Test-Path -LiteralPath $out){throw 'Snapshot exists; no repeat authorized.'}
$code=Get-Content -Raw -LiteralPath 'D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\T072N-work\research_log\T072N\snapshot.py'
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($code))
$command="/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -c `"import base64;exec(base64.b64decode('$encoded'))`""
$raw=Invoke-AutodlSsh $command
$raw | Set-Content -LiteralPath $out -Encoding utf8
$raw
