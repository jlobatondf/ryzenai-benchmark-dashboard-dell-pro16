#!/bin/bash
# Setup script for Digital Ocean Ubuntu 22.04 droplet
# Run as root: bash setup-droplet.sh

set -e

echo "=== Dell Pro 16 Dashboard Droplet Setup ==="

# Update system
echo "Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.11
echo "Installing Python 3.11..."
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install nginx
echo "Installing nginx..."
apt install -y nginx

# Install certbot for SSL
echo "Installing certbot..."
apt install -y certbot python3-certbot-nginx

# Create app directory
echo "Creating application directory..."
mkdir -p /opt/dell-pro16-dashboard
chown -R root:root /opt/dell-pro16-dashboard

# Create log directory
mkdir -p /var/log/dell-pro16

# Configure firewall
echo "Configuring firewall..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Clone your repo: git clone <repo-url> /opt/dell-pro16-dashboard"
echo "2. cd /opt/dell-pro16-dashboard"
echo "3. Create venv: python3.11 -m venv venv"
echo "4. Activate: source venv/bin/activate"
echo "5. Install deps: pip install -r requirements.txt"
echo "6. Create .env: cp .env.example .env && nano .env"
echo "7. Copy nginx config: cp deploy/nginx.conf /etc/nginx/sites-available/dell-pro16"
echo "8. Enable nginx: ln -s /etc/nginx/sites-available/dell-pro16 /etc/nginx/sites-enabled/"
echo "9. Remove default: rm -f /etc/nginx/sites-enabled/default"
echo "10. Test nginx: nginx -t"
echo "11. Get SSL: certbot --nginx -d your-subdomain.yourdomain.com"
echo "12. Copy services: cp deploy/*.service /etc/systemd/system/"
echo "13. Reload systemd: systemctl daemon-reload"
echo "14. Start services: systemctl enable --now dell-pro16-dashboard dell-pro16-chatbot"
echo "15. Reload nginx: systemctl reload nginx"
