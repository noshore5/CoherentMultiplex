# DigitalOcean Deployment Troubleshooting

## Problem: "Error: undefined" on DigitalOcean but works locally

This typically means DigitalOcean App Platform is not using your latest code. Here are the solutions:

### 🔍 **Step 1: Verify Current Deployment**

After deployment, check these URLs:
- `https://your-app.ondigitalocean.app/version` - Shows build timestamp
- `https://your-app.ondigitalocean.app/health` - Shows Claude AI status

### 🔄 **Step 2: Force DigitalOcean to Use Latest Code**

#### Option A: Force Rebuild (Fastest)
1. Go to DigitalOcean App Platform dashboard
2. Select your app → Settings → General
3. Click "Force Rebuild" button
4. Wait for deployment to complete

#### Option B: Git Push Method
```bash
# Make sure all changes are committed
git add .
git commit -m "fix: force deployment rebuild"
git push origin main

# Then force rebuild in DO dashboard
```

#### Option C: Nuclear Option (If nothing else works)
1. Delete the app completely in DigitalOcean
2. Create a new app from your GitHub repo
3. Set environment variables again

### 🏥 **Step 3: Common Issues & Fixes**

#### Environment Variables Not Set:
- Go to App Platform → Settings → App-Level Environment Variables
- Add: `CLAUDE_API_KEY` = `your-api-key` (as SECRET)

#### Build Cache Issues:
- DigitalOcean sometimes caches old builds
- Force rebuild usually fixes this

#### Git Branch Issues:
- Make sure DO is tracking the correct branch (usually `main`)
- Check: App Settings → Source

### 🧪 **Step 4: Verification Commands**

#### Local Test:
```bash
python do_deploy_check.py  # Run before deployment
```

#### Remote Test (after deployment):
```bash
curl https://your-app.ondigitalocean.app/version
curl https://your-app.ondigitalocean.app/health
```

### 📝 **Step 5: Logs & Debugging**

#### Check DigitalOcean Logs:
1. Go to your app dashboard
2. Click "Runtime Logs" tab
3. Look for Python errors or Claude API issues

#### Common Log Messages:
- `✓ Using Claude API key from environment variable` ✅ Good
- `⚠️ No Claude API key found` ❌ Environment variable not set
- `❌ Claude AI: Not available` ❌ Import/config error

### 🚨 **Emergency Deployment Steps**

If you need to get it working immediately:

1. **Quick Fix Method:**
   ```bash
   # Update version info to force cache bust
   python do_deploy_check.py
   git add version_info.json
   git commit -m "cache bust: $(date)"
   git push origin main
   ```

2. **Go to DO dashboard and Force Rebuild**

3. **Check deployment:**
   ```bash
   curl https://your-app.ondigitalocean.app/version
   ```

### 📊 **Expected Response from /version endpoint:**
```json
{
  "build_time": "2025-08-11T...",
  "timestamp": 1723377600,
  "claude_utils_version": "v2.0_fixed",
  "deployment_check": true
}
```

If timestamp is old or missing, DO is using cached/old code.

### 🔧 **Why This Happens:**

1. **DigitalOcean Build Cache:** DO caches Docker layers and sometimes doesn't detect changes
2. **Git Detection Issues:** Sometimes DO doesn't pull the latest commit
3. **Environment Differences:** Local vs production environment variables
4. **File System Differences:** Case sensitivity, path differences

The version endpoint we added will help you immediately identify if DO is using your latest code!
