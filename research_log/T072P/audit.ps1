$ErrorActionPreference='Stop'
$env:AUTODL_CONFIG_PATH='D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\config.json'
. 'D:\work\claude-autodl\autodl-workflow-clean\scripts\Autodl.Common.ps1'
$folder='D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\T072P-work\research_log\T072P'
$out=Join-Path $folder 'receipt.json'
if(Test-Path -LiteralPath $out){throw 'Audit already recorded; do not repeat.'}
$source=Get-Content -Raw -LiteralPath (Join-Path $folder 'audit.py')
$manifest=[Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $folder 'manifest.json')))
$source += "`nimport base64`nm=json.loads(base64.b64decode('$manifest'))`nh=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()`nprint(json.dumps(run(m,h),indent=2))`n"
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($source))
$command="/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -c `"import base64;exec(base64.b64decode('$encoded'))`""
$raw=Invoke-AutodlSsh $command
$raw | Set-Content -LiteralPath $out -Encoding utf8
$raw
