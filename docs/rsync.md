# rsync procedures for dev -> prod and prod -> dev.

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
    joe@euler:~/prod/ai-chat/ ~/dev/ai-chat/
```
