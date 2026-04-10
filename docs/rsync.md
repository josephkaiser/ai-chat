# rsync procedures for dev -> prod and prod -> dev.

Managed chat Python environments now live outside the repo by default, so new package installs should not bloat `runs/` syncs. The legacy `runs/*/python-env/` directories can still exist on older runs, so the prod -> dev sync below excludes them explicitly. Hidden dotfiles may still exist inside workspaces for internal state, but the browser now hides dot-prefixed paths unless explicitly targeted.

## dev -> prod
excludes venv and node downloaded files

```bash
rsync -avz --progress \
    --exclude='.venv/' \
    --exclude='.git/' \
    --exclude='node_modules/' \
    --exclude='.DS_Store' \
    --exclude='.pycache/' \
    --exclude='.pycache_compile/' \
    --exclude='feeds/' \
    --exclude='logs/' \
    --exclude='data/' \
    --exclude='runs/' \
    ~/dev/ai-chat/ joe@euler:~/prod/ai-chat/
```

## prod -> dev
includes data and logs

```bash
rsync -avz --progress \
    --exclude='.venv/' \
    --exclude='.git/' \
    --exclude='node_modules/' \
    --exclude='.DS_Store' \
    --exclude='.pycache/' \
    --exclude='.pycache_compile/' \
    --exclude='runs/*/python-env/' \
    joe@euler:~/prod/ai-chat/ ~/dev/ai-chat/
```
