# Performance Improvements & Optimizations

## Critical Performance Issues

### 1. **Database Connection Management** ⚠️ HIGH IMPACT
**Problem**: Every database operation opens/closes a new connection. This is very inefficient.

**Current**: 20+ `sqlite3.connect()` calls per request cycle
**Impact**: High latency, file I/O overhead, connection overhead

**Solution Options**:
- **Option A (Recommended)**: Use `aiosqlite` for async database operations
  - Tradeoff: Requires async refactoring, but eliminates blocking
  - Performance gain: 3-5x faster database operations
  - Code changes: Moderate

- **Option B (Simpler)**: Connection pooling with `sqlite3.connect()` + WAL mode
  - Tradeoff: Still blocking, but much faster
  - Performance gain: 2-3x faster, better concurrency
  - Code changes: Minimal

- **Option C (Quick fix)**: Enable WAL mode + connection reuse
  - Tradeoff: Minimal changes
  - Performance gain: 1.5-2x faster
  - Code changes: Very minimal

**Recommended**: Option B (WAL mode + connection reuse) for quick wins, then migrate to Option A

---

### 2. **SQLite WAL Mode Not Enabled** ⚠️ HIGH IMPACT
**Problem**: Default journal mode is slower for concurrent reads

**Fix**: Add to `init_db()`:
```python
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA synchronous=NORMAL')  # Faster than FULL, still safe
conn.execute('PRAGMA cache_size=-64000')  # 64MB cache (adjust based on RAM)
conn.execute('PRAGMA temp_store=MEMORY')  # Use RAM for temp tables
```

**Performance gain**: 2-3x faster concurrent reads, better write performance
**Tradeoff**: Slightly more complex recovery (but WAL is actually safer)

---

### 3. **Unbounded Log Capture Memory** ⚠️ MEDIUM IMPACT
**Problem**: `log_capture = io.StringIO()` grows unbounded, can consume GBs of RAM

**Current**: Logs accumulate forever in memory
**Impact**: Memory leak over time, especially with verbose logging

**Solution**: Implement circular buffer or size limit
```python
from collections import deque
MAX_LOG_SIZE = 100000  # ~100KB of logs
log_buffer = deque(maxlen=MAX_LOG_SIZE)
```

**Performance gain**: Prevents memory leaks, stable memory usage
**Tradeoff**: Older logs are lost (but they're in stdout anyway)

---

### 4. **CSS Generated on Every Request** ⚠️ MEDIUM IMPACT
**Problem**: `generate_css()` runs on every page load, generating 2000+ lines of CSS

**Current**: CSS regenerated for every `/` request
**Impact**: CPU waste, unnecessary string operations

**Solution**: Cache CSS by mode
```python
from functools import lru_cache

@lru_cache(maxsize=2)  # Cache light and dark modes
def generate_css(mode='light'):
    # ... existing code
```

**Performance gain**: 10-20ms saved per request, reduces CPU usage
**Tradeoff**: Minimal memory overhead (negligible)

---

### 5. **Blocking Database Operations in Async Context** ⚠️ MEDIUM IMPACT
**Problem**: All database operations block the event loop

**Current**: `save_message()`, `get_conversation_history()` are blocking
**Impact**: Blocks other requests while waiting for DB I/O

**Solution**: Use `asyncio.to_thread()` or `aiosqlite`
```python
async def save_message_async(conv_id, role, content):
    return await asyncio.to_thread(save_message, conv_id, role, content)
```

**Performance gain**: Better concurrency, non-blocking I/O
**Tradeoff**: Slightly more complex code

---

### 6. **Inefficient Token Estimation** ⚠️ LOW-MEDIUM IMPACT
**Problem**: Token estimation uses rough `/4` division, loads all messages into memory

**Current**: `len(msg['content']) // 4` is inaccurate, loads entire conversation
**Impact**: Inaccurate token counting, unnecessary memory usage

**Solution**: Use `tiktoken` library for accurate counting, or at least improve estimation
```python
# Better estimation: account for message overhead
def estimate_tokens(text):
    # Rough: 1 token ≈ 4 chars, but add overhead for message structure
    return (len(text) // 4) + 20  # +20 for message formatting overhead
```

**Performance gain**: More accurate context window usage, better model performance
**Tradeoff**: Slightly more CPU (negligible)

---

## Docker & Infrastructure Optimizations

### 7. **Multi-Stage Docker Build** ⚠️ LOW IMPACT (Build time only)
**Problem**: Final image includes build tools and dependencies

**Solution**: Multi-stage build to reduce image size
```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app.py theme_config.py .
ENV PATH=/root/.local/bin:$PATH
```

**Performance gain**: Smaller image (50-70% reduction), faster pulls
**Tradeoff**: Slightly more complex Dockerfile

---

### 8. **Python Optimizations** ⚠️ LOW IMPACT
**Problem**: Python runs in interpreted mode

**Solution**: Add Python optimizations to Dockerfile
```dockerfile
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=1  # Remove assert statements, optimize bytecode
```

**Performance gain**: 5-10% faster execution, less memory
**Tradeoff**: No assert statements (usually fine for production)

---

### 9. **Uvicorn Workers** ⚠️ MEDIUM IMPACT (if multi-core)
**Problem**: Single worker handles all requests

**Solution**: Use multiple workers (if multi-core CPU available)
```dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Performance gain**: 2-4x throughput on multi-core systems
**Tradeoff**: More memory usage, need to handle shared state carefully

---

## Code-Level Optimizations

### 10. **Conversation History Caching** ⚠️ MEDIUM IMPACT
**Problem**: Conversation history loaded from DB on every request

**Solution**: Cache recent conversations in memory
```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=100)
def get_cached_conversation(conv_id, cache_key):
    # cache_key changes when conversation is updated
    return get_conversation_history(conv_id)
```

**Performance gain**: 10-50ms saved per request for cached conversations
**Tradeoff**: Memory usage (but conversations are small)

---

### 11. **WebSocket Log Polling Optimization** ⚠️ LOW IMPACT
**Problem**: Polls every 0.5s even when no new logs

**Current**: `await asyncio.sleep(0.5)` always runs
**Solution**: Adaptive polling (faster when logs are active, slower when idle)
```python
poll_interval = 0.1  # Start fast
while True:
    await asyncio.sleep(poll_interval)
    # If no new logs for 5s, increase interval to 2s
    # If logs detected, reset to 0.1s
```

**Performance gain**: Reduces CPU when idle
**Tradeoff**: Slightly more complex logic

---

### 12. **Database Query Optimization** ⚠️ MEDIUM IMPACT
**Problem**: Some queries could use better indexes or be optimized

**Current Issues**:
- `search_messages()` uses `LIKE` which can't use indexes efficiently
- Conversation loading could batch better

**Solutions**:
- Add full-text search index for message content (FTS5)
- Use prepared statements (already doing this, good!)
- Consider pagination for large result sets

**Performance gain**: 2-5x faster searches
**Tradeoff**: More complex setup, larger database size

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Enable SQLite WAL mode (#2)
2. ✅ Cache CSS generation (#4)
3. ✅ Limit log capture size (#3)
4. ✅ Add Python optimizations (#8)

### Phase 2: Medium Effort (4-6 hours)
5. ✅ Connection pooling or aiosqlite (#1)
6. ✅ Wrap DB calls in `asyncio.to_thread()` (#5)
7. ✅ Improve token estimation (#6)
8. ✅ Conversation caching (#10)

### Phase 3: Advanced (8+ hours)
9. ✅ Multi-stage Docker build (#7)
10. ✅ Full-text search (#12)
11. ✅ Uvicorn workers (#9)
12. ✅ Adaptive log polling (#11)

---

## Performance Impact Summary

| Optimization | Impact | Effort | Tradeoff |
|-------------|--------|--------|----------|
| SQLite WAL Mode | 🔥🔥🔥 High | ⚡ Low | Minimal |
| Connection Pooling | 🔥🔥🔥 High | ⚡⚡ Medium | Low |
| Log Memory Limit | 🔥🔥 Medium | ⚡ Low | Old logs lost |
| CSS Caching | 🔥 Low | ⚡ Low | None |
| Async DB Operations | 🔥🔥 Medium | ⚡⚡ Medium | Code complexity |
| Token Estimation | 🔥 Low | ⚡ Low | None |
| Multi-stage Build | 🔥 Low | ⚡⚡ Medium | Build complexity |
| Python Optimize | 🔥 Low | ⚡ Low | No asserts |
| Uvicorn Workers | 🔥🔥 Medium | ⚡⚡ Medium | Memory usage |
| Conversation Cache | 🔥 Low | ⚡ Low | Memory usage |
| FTS Search | 🔥🔥 Medium | ⚡⚡⚡ High | DB size |

**Total Estimated Performance Gain**: 3-5x faster database operations, 20-30% overall improvement

---

## Notes on Tradeoffs

- **WAL Mode**: Slightly more complex recovery, but actually safer and faster
- **Connection Pooling**: More memory, but much faster
- **Log Limits**: Lose old logs, but prevent memory leaks
- **Async DB**: More complex code, but non-blocking
- **Workers**: More memory, but better throughput
- **Caching**: Memory usage, but faster responses

All tradeoffs are reasonable for a production system focused on performance.

