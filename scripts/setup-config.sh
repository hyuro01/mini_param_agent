#!/bin/bash
# mini_param_agent Configuration Setup Script
# Copies configuration templates from this project checkout.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration directory
CONFIG_DIR="$HOME/.mini_param_agent/config"

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   mini_param_agent Configuration Setup        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Create config directory
echo -e "${BLUE}[1/2]${NC} Creating configuration directory..."
if [ -d "$CONFIG_DIR" ]; then
    # Auto backup existing config
    BACKUP_DIR="$HOME/.mini_param_agent/config.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}   Configuration directory exists, backing up to:${NC}"
    echo -e "${YELLOW}   $BACKUP_DIR${NC}"
    cp -r "$CONFIG_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}   ✓ Backup created${NC}"
else
    mkdir -p "$CONFIG_DIR"
    echo -e "${GREEN}   ✓ Created: $CONFIG_DIR${NC}"
fi

# Step 2: Copy configuration files from this checkout
echo -e "${BLUE}[2/2]${NC} Copying configuration files..."

FILES_COPIED=0
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/../mini_param_agent/config"

# Copy config-example.yaml as config.yaml
if cp "$TEMPLATE_DIR/config-example.yaml" "$CONFIG_DIR/config.yaml" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Copied: config.yaml${NC}"
    FILES_COPIED=$((FILES_COPIED + 1))
else
    echo -e "${RED}   ✗ Failed to copy: config.yaml${NC}"
fi

# Copy mcp-example.json as mcp.json (optional, user should customize)
if cp "$TEMPLATE_DIR/mcp-example.json" "$CONFIG_DIR/mcp.json" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Copied: mcp.json (from template)${NC}"
    FILES_COPIED=$((FILES_COPIED + 1))
fi

# Copy system_prompt.md (optional)
if cp "$TEMPLATE_DIR/system_prompt.md" "$CONFIG_DIR/system_prompt.md" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Copied: system_prompt.md${NC}"
    FILES_COPIED=$((FILES_COPIED + 1))
fi

if [ $FILES_COPIED -eq 0 ]; then
    echo -e "${RED}   ✗ Failed to copy configuration files${NC}"
    echo -e "${YELLOW}   Please run this script from the project checkout${NC}"
    exit 1
fi

echo -e "${GREEN}   ✓ Configuration files ready${NC}"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup Complete! ✨                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Configuration files location:"
echo -e "  ${CYAN}$CONFIG_DIR${NC}"
echo ""
echo -e "Files:"
ls -1 "$CONFIG_DIR" 2>/dev/null | sed 's/^/  📄 /' || echo "  (no files yet)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo -e "${YELLOW}1. Install mini_param_agent from its project checkout:${NC}"
echo -e "   ${GREEN}uv tool install --editable '.[ml]'${NC}"
echo ""
echo -e "${YELLOW}2. Configure your API Key:${NC}"
echo -e "   Edit config.yaml and add your MiniMax API Key:"
echo -e "   ${GREEN}nano $CONFIG_DIR/config.yaml${NC}"
echo -e "   ${GREEN}vim $CONFIG_DIR/config.yaml${NC}"
echo -e "   ${GREEN}code $CONFIG_DIR/config.yaml${NC}"
echo ""
echo -e "${YELLOW}3. Start using mini_param_agent:${NC}"
echo -e "   ${GREEN}mini_param_agent${NC}                              # Use current directory"
echo -e "   ${GREEN}mini_param_agent --workspace /path/to/project${NC} # Specify workspace"
echo -e "   ${GREEN}mini_param_agent --help${NC}                      # Show help"
echo ""
