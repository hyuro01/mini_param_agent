# mini_param_agent Configuration Setup Script for Windows
# Copies configuration templates from this project checkout.

# Error handling
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )

    $colorMap = @{
        "Red" = [ConsoleColor]::Red
        "Green" = [ConsoleColor]::Green
        "Yellow" = [ConsoleColor]::Yellow
        "Blue" = [ConsoleColor]::Blue
        "Cyan" = [ConsoleColor]::Cyan
        "White" = [ConsoleColor]::White
    }

    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

# Configuration directory
$CONFIG_DIR = Join-Path $env:USERPROFILE ".mini_param_agent\config"

Write-ColorOutput "==================================================" -Color "Cyan"
Write-ColorOutput "   mini_param_agent Configuration Setup" -Color "Cyan"
Write-ColorOutput "==================================================" -Color "Cyan"
Write-Host ""

# Step 1: Create config directory
Write-ColorOutput "[1/2] Creating configuration directory..." -Color "Blue"

if (Test-Path $CONFIG_DIR) {
    # Auto backup existing config
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BACKUP_DIR = Join-Path $env:USERPROFILE ".mini_param_agent\config.backup.$timestamp"
    Write-ColorOutput "   Configuration directory exists, backing up to:" -Color "Yellow"
    Write-ColorOutput "   $BACKUP_DIR" -Color "Yellow"
    Copy-Item -Path $CONFIG_DIR -Destination $BACKUP_DIR -Recurse
    Write-ColorOutput "   [OK] Backup created" -Color "Green"
} else {
    New-Item -Path $CONFIG_DIR -ItemType Directory -Force | Out-Null
    Write-ColorOutput "   [OK] Created: $CONFIG_DIR" -Color "Green"
}

# Step 2: Copy configuration files from this checkout
Write-ColorOutput "[2/2] Copying configuration files..." -Color "Blue"

$FILES_COPIED = 0
$TEMPLATE_DIR = Join-Path $PSScriptRoot "..\mini_param_agent\config"

# Copy config-example.yaml as config.yaml
try {
    $configPath = Join-Path $CONFIG_DIR "config.yaml"
    Copy-Item -Path (Join-Path $TEMPLATE_DIR "config-example.yaml") -Destination $configPath -Force
    Write-ColorOutput "   [OK] Copied: config.yaml" -Color "Green"
    $FILES_COPIED++
} catch {
    Write-ColorOutput "   [ERROR] Failed to copy: config.yaml" -Color "Red"
}

# Copy mcp-example.json as mcp.json
try {
    $mcpPath = Join-Path $CONFIG_DIR "mcp.json"
    Copy-Item -Path (Join-Path $TEMPLATE_DIR "mcp-example.json") -Destination $mcpPath -Force
    Write-ColorOutput "   [OK] Copied: mcp.json" -Color "Green"
    $FILES_COPIED++
} catch {
    # Optional file
}

# Copy system_prompt.md
try {
    $promptPath = Join-Path $CONFIG_DIR "system_prompt.md"
    Copy-Item -Path (Join-Path $TEMPLATE_DIR "system_prompt.md") -Destination $promptPath -Force
    Write-ColorOutput "   [OK] Copied: system_prompt.md" -Color "Green"
    $FILES_COPIED++
} catch {
    # Optional file
}

if ($FILES_COPIED -eq 0) {
    Write-ColorOutput "   [ERROR] Failed to copy configuration files" -Color "Red"
    Write-ColorOutput "   Please run this script from the project checkout" -Color "Yellow"
    exit 1
}

Write-ColorOutput "   [OK] Configuration files ready" -Color "Green"

Write-Host ""
Write-ColorOutput "==================================================" -Color "Green"
Write-ColorOutput "   Setup Complete!" -Color "Green"
Write-ColorOutput "==================================================" -Color "Green"
Write-Host ""
Write-Host "Configuration files location:"
Write-ColorOutput "  $CONFIG_DIR" -Color "Cyan"
Write-Host ""
Write-Host "Files:"
Get-ChildItem $CONFIG_DIR -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "   $($_.Name)"
}
Write-Host ""
Write-ColorOutput "Next Steps:" -Color "Yellow"
Write-Host ""
Write-ColorOutput "1. Install mini_param_agent from its project checkout:" -Color "Yellow"
Write-ColorOutput "   uv tool install --editable '.[ml]'" -Color "Green"
Write-Host ""
Write-ColorOutput "2. Configure your API Key:" -Color "Yellow"
Write-Host "   Edit config.yaml and add your MiniMax API Key:"
Write-ColorOutput "   notepad $CONFIG_DIR\config.yaml" -Color "Green"
Write-ColorOutput "   code $CONFIG_DIR\config.yaml" -Color "Green"
Write-Host ""
Write-ColorOutput "3. Start using mini_param_agent:" -Color "Yellow"
Write-ColorOutput "   mini_param_agent                              # Use current directory" -Color "Green"
Write-ColorOutput "   mini_param_agent --workspace C:\path\to\project # Specify workspace" -Color "Green"
Write-ColorOutput "   mini_param_agent --help                      # Show help" -Color "Green"
Write-Host ""
