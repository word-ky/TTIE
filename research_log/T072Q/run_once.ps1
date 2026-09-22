$ErrorActionPreference='Stop'
$env:AUTODL_CONFIG_PATH='D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\config.json'
. 'D:\work\claude-autodl\autodl-workflow-clean\scripts\Autodl.Common.ps1'
$release='/home/wenchang/asdasdsad/wjq/TTIE/releases/20260923-050852-ttie-t072q-sealed-smoke'
$check=@'
import hashlib,json
from pathlib import Path
p=Path('research_log/T072O');s=json.loads((p/'seal.json').read_bytes())
assert s['root_sha256']=='c000543fc2623f76d1270d7c479114ea325ad3971d12821bbf46bcd8629e8c9d'
assert hashlib.sha256(json.dumps(s['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==s['root_sha256']
for n,h in s['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
print('SEALED_REMOTE_SOURCE_PASS')
'@
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($check))
Invoke-AutodlSsh "cd '$release' && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -c `"import base64;exec(base64.b64decode('$encoded'))`""
$command="cd '$release' && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T072O/launcher.py --runtime-root /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines --out /media/wenchang/F/wjq/TTIE/runs/T072Q-sealed-smoke-once"
& 'D:\work\claude-autodl\autodl-workflow-clean\scripts\autodl-run.ps1' -Name ttie-t072q-sealed-smoke -Cmd $command
