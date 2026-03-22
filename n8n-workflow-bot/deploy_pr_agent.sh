#!/usr/bin/env bash
# deploy_pr_agent.sh — Deploy the PR Attachment Validation Agent to n8n
set -e
cd "$(dirname "$0")/n8n-workflow-bot" 2>/dev/null || cd "$(dirname "$0")"

echo "================================================================"
echo "  PR Attachment Validation Agent — Deployment"
echo "================================================================"
echo ""

# Check .env exists
if [ ! -f .env ]; then
  echo "ERROR: .env file not found. Copy .env.example to .env and add your keys."
  exit 1
fi

# Check Python deps
python3 -c "import requests, pydantic, dotenv" 2>/dev/null || {
  echo "Installing Python dependencies..."
  pip install -r requirements.txt -q
}

echo "Step 1/2 — Deploying Drive Setup Workflow..."
python3 main.py --json workflows/pr-drive-setup-workflow.json
echo ""

echo "Step 2/2 — Deploying PR Validation Agent Workflow..."
python3 main.py --json workflows/pr-validation-workflow.json
echo ""

echo "================================================================"
echo "  DEPLOYMENT COMPLETE"
echo "================================================================"
echo ""
echo "  Form:     Open n8n-workflow-bot/pr-form.html in a browser"
echo "            (or serve via: python3 -m http.server 8080)"
echo ""
echo "  IMPORTANT — One-time Drive setup:"
echo "  1. Open your n8n instance: https://ashleymartian.app.n8n.cloud"
echo "  2. Run the 'PR Drive Folder Setup' workflow manually once"
echo "  3. Copy the folder ID from the execution output"
echo "  4. Go to n8n Settings → Variables → add:"
echo "       Name:  DRIVE_DRAWINGS_FOLDER_ID"
echo "       Value: <the folder ID you copied>"
echo ""
echo "  Webhook URL (for form): https://ashleymartian.app.n8n.cloud/webhook/pr-validation"
echo "================================================================"
