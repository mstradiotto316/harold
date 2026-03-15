# 2026-03-15 Session 48 (desktop)

## Summary
- Removed a publicly committed Wi-Fi credential from the repository history.
- Purged the tracked NetworkManager profile from `main` and `autoresearch/session-2026-03-14`.
- Preserved the local machine's `.nmconnection` file as an ignored, untracked file.
- Restored the repo remote and re-pushed the rewritten public branches with lease checks.

## Remediation
- Located the leak in the tracked root-level `.nmconnection` file.
- Confirmed the profile contained a real Wi-Fi PSK and that the file existed in public branch history.
- Backed up the repo state to `/tmp/harold-secret-cleanup.1ooni1/` before rewriting:
  - `repo.bundle`
  - `working-tree.patch`
  - local `.nmconnection` copy
  - remote metadata
- Ran `git filter-repo --path <profile>.nmconnection --invert-paths` to remove the file from history.
- Ran `git filter-repo --replace-text` to redact historical Wi-Fi identifier strings left in docs.
- Re-added `origin` after the rewrite and force-pushed:
  - `main`
  - `autoresearch/session-2026-03-14`

## Verification
- Verified `HEAD` no longer contains the tracked `.nmconnection` path.
- Verified exact-history searches for the former PSK returned no matches.
- Verified exact-history searches for the former SSID returned no matches after the second rewrite.
- Verified the remote branch heads moved to the rewritten commits:
  - `origin/main` -> `ede4994bb256646eb93d66d38f01473fdaa4749d`
  - `origin/autoresearch/session-2026-03-14` -> `2821a0e5553bb48566d3aa30a3bd8361bfe15c38`

## Notes
- Another agent committed `2821a0e` on `autoresearch/session-2026-03-14` during remediation; the final force-push included that commit.
- The only remaining operational follow-up is credential rotation, since public history exposure should be treated as compromise.
