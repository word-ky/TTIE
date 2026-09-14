$env:AUTODL_CONFIG_PATH='D:\work\fightccfa-agin\CVPR2027\TTT-ImageEnhancement\.autodl\config.json'
. 'D:\work\claude-autodl\autodl-workflow-clean\scripts\Autodl.Common.ps1'
$remote=@'
set -e
cd /home/wenchang/asdasdsad/wjq/TTIE
mkdir -p shared/t025a /media/wenchang/F/wjq/TTIE/shared/t025a
test -f runs/20260914-075823-ttie-t025a-oracle/artifacts/REFERENCE_ORACLE_ONLY/independent_aggregation.json
tar -cf /media/wenchang/F/wjq/TTIE/shared/t025a/T025A_execution.tar runs/20260914-075823-ttie-t025a-oracle releases/20260914-075755-ttie-t025a-oracle
tar --exclude=oracle_output.pt -czf shared/t025a/T025A_compact.tar.gz runs/20260914-075823-ttie-t025a-oracle
sha256sum /media/wenchang/F/wjq/TTIE/shared/t025a/T025A_execution.tar shared/t025a/T025A_compact.tar.gz
'@
Invoke-AutodlSsh -Command $remote.Replace("`r`n","`n")
$base=Get-AutodlSshBaseArgs
$scpArgs=@($base.Common)+@('-O','-P',[string]$base.Config.port,"$($base.Target):/home/wenchang/asdasdsad/wjq/TTIE/shared/t025a/T025A_compact.tar.gz",(Join-Path $PWD 'research_log\T025A_compact.tar.gz'))
& scp @scpArgs
if($LASTEXITCODE -ne 0){throw 'T025A compact fetch failed'}
