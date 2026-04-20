# Setup Guide - Agentic RAG System

## Critical: API Key Configuration

### The Problem
The error `"Missing Authentication header"` occurs when the `OPENROUTER_API_KEY` is not properly passed to the application.

### The Solution

#### Option 1: Using .env File (RECOMMENDED)

1. **Create/Edit `.env` file** in the project root:
   ```bash
   cd /workspace
   cp .env.example .env
   ```

2. **Add your REAL OpenRouter API key**:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-YOUR-ACTUAL-KEY-HERE
   ```
   
   ⚠️ **IMPORTANT**: Replace `sk-or-v1-YOUR-ACTUAL-KEY-HERE` with your real key from https://openrouter.ai/keys

3. **Verify .env is in .gitignore**:
   ```bash
   cat .gitignore | grep ".env"
   # Should show: .env
   ```

4. **Restart Docker containers**:
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

#### Option 2: Environment Variable (for testing)

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-real-key"
docker-compose up --build -d
```

#### Option 3: In-Memory Mode (No Docker Required)

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-real-key"
export QDRANT_MODE=memory
pip install -r requirements.txt
uvicorn agentic_rag.api.server:app --host 0.0.0.0 --port 8000
```

## Verification Steps

### 1. Check .env file exists and has correct format
```bash
cat /workspace/.env
# Should show OPENROUTER_API_KEY=sk-or-v1-...
```

### 2. Check .gitignore includes .env
```bash
grep "^\.env$" /workspace/.gitignore
# Should output: .env
```

### 3. Test configuration loading
```bash
cd /workspace
python -c "from agentic_rag.config import settings; print('API Key:', settings.openrouter_api_key[:15] + '...' if settings.openrouter_api_key else 'MISSING')"
```

### 4. Check Docker logs for API key initialization
```bash
docker-compose logs api | grep "Initializing OpenRouter"
# Should show: Initializing OpenRouter embedder with key: sk-or-...
```

### 5. Test health endpoint
```bash
curl http://localhost:8000/health
```

## Common Issues

### Issue: "Missing Authentication header"
**Cause**: API key not set or invalid

**Fix**:
1. Verify `.env` file exists: `ls -la /workspace/.env`
2. Check API key format: `cat /workspace/.env | grep OPENROUTER`
3. Rebuild containers: `docker-compose down && docker-compose up --build -d`
4. Get a new key if needed: https://openrouter.ai/keys

### Issue: Docker won't start
**Fix**: Use in-memory mode
```bash
export OPENROUTER_API_KEY="your-key"
export QDRANT_MODE=memory
uvicorn agentic_rag.api.server:app --host 0.0.0.0 --port 8000
```

### Issue: .env file being committed to git
**Fix**: 
```bash
git rm --cached .env
echo ".env" >> .gitignore
git commit -m "Remove .env from tracking"
```

## File Upload Example

Once everything is working:

```bash
# Upload a text file
curl -X POST "http://localhost:8000/files/upload" \
  -F "file=@test.txt" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"

# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

## Security Notes

1. **NEVER commit `.env` to git** - it's in `.gitignore` for a reason
2. **Rotate keys regularly** - get new keys from https://openrouter.ai/keys
3. **Use different keys for dev/prod** - don't share keys between environments
4. **Check file permissions**: `chmod 600 .env`

## Next Steps

1. Get your OpenRouter API key: https://openrouter.ai/keys
2. Add it to `.env`
3. Run: `docker-compose up --build -d`
4. Test: `curl http://localhost:8000/health`
5. Upload files and start querying!
