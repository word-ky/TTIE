$ErrorActionPreference = 'Stop'
$env:AUTODL_CONFIG_PATH = 'D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\config.json'
. 'D:\work\claude-autodl\autodl-workflow-clean\scripts\Autodl.Common.ps1'
$receiptPath = 'D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\research_log\T072M_gate_raw.txt'
if (Test-Path -LiteralPath $receiptPath) { throw 'Snapshot already exists; no repeat gate authorized.' }
$command = 'date -u +%Y-%m-%dT%H:%M:%SZ; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits'
$raw = Invoke-AutodlSsh $command
$raw | Set-Content -LiteralPath $receiptPath -Encoding utf8
$raw
