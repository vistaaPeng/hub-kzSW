# 以管理员身份运行 PowerShell，启用 WSL2 并安装 Ubuntu 22.04
# 执行后重启，再在 Ubuntu 中运行 setup_wsl_inside.sh

Write-Host "=== work9: vLLM 作业 WSL2 环境准备 ===" -ForegroundColor Cyan

$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName "Microsoft-Windows-Subsystem-Linux"
$vmFeature = Get-WindowsOptionalFeature -Online -FeatureName "VirtualMachinePlatform"

if ($wslFeature.State -ne "Enabled") {
    Write-Host "启用 WSL..." -ForegroundColor Yellow
    Enable-WindowsOptionalFeature -Online -FeatureName "Microsoft-Windows-Subsystem-Linux" -All -NoRestart
}
if ($vmFeature.State -ne "Enabled") {
    Write-Host "启用虚拟机平台..." -ForegroundColor Yellow
    Enable-WindowsOptionalFeature -Online -FeatureName "VirtualMachinePlatform" -All -NoRestart
}

wsl --set-default-version 2
wsl --install -d Ubuntu-22.04 --no-launch

Write-Host ""
Write-Host "请重启电脑。重启后打开 Ubuntu，执行：" -ForegroundColor Green
Write-Host '  bash "/mnt/e/DeepLearning/week9/week9 大模型应用补充知识/work9/setup_wsl_inside.sh"'
