# 启动火山引擎LLM Agent服务
# 使用方法：在Anaconda PowerShell Prompt中运行此脚本

Write-Host "🌋 启动火山引擎LLM Agent服务..." -ForegroundColor Green

# 获取Anaconda初始化脚本路径
$condaPath = $env:CONDA_PREFIX
if (-not $condaPath) {
    Write-Host "❌ 错误：未找到Conda环境，请使用Anaconda PowerShell Prompt" -ForegroundColor Red
    exit 1
}

# 初始化Conda（如果尚未初始化）
$condaInit = ""
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $condaInit = "& '$env:CONDA_EXE' shell.powershell hook | Out-String | Invoke-Expression; "
}

# 启动火山引擎LLM Agent
Write-Host "[1/1] 启动火山引擎LLM Agent (env: wakefusion)..." -ForegroundColor Green
$llmCmd = "${condaInit}conda activate wakefusion; python -m wakefusion.services.llm_agent_volcano"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $llmCmd -WindowStyle Normal

Write-Host ""
Write-Host "✅ 火山引擎LLM Agent服务已启动" -ForegroundColor Green
Write-Host ""
Write-Host "服务信息：" -ForegroundColor Yellow
Write-Host "  - LLM Agent WebSocket: ws://127.0.0.1:8080/api/voice/ws"
Write-Host "  - 火山引擎API: https://ark.cn-beijing.volces.com/api/v3/responses"
Write-Host ""
Write-Host "⚠️  注意：" -ForegroundColor Yellow
Write-Host "  1. 请确保 config/config.yaml 中的 volcano_api_key 已配置为你的API密钥"
Write-Host "  2. 启动后，可以使用 launcher.ps1 启动WakeFusion系统"
Write-Host "  3. Core Server 会自动连接到LLM Agent"
Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
