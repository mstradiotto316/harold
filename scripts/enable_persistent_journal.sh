#!/usr/bin/env bash
set -euo pipefail

# Enable persistent systemd journal storage for post-crash diagnostics.
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal

sudo mkdir -p /etc/systemd/journald.conf.d
cat <<'EOF' | sudo tee /etc/systemd/journald.conf.d/harold.conf >/dev/null
[Journal]
Storage=persistent
SystemMaxUse=200M
RuntimeMaxUse=50M
EOF

sudo systemctl restart systemd-journald

echo "Persistent journaling enabled."
