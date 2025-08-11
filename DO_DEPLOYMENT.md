# DigitalOcean Deployment Guide

## Setting Environment Variables on DigitalOcean

### For App Platform:

1. Go to your App Platform dashboard
2. Select your app
3. Go to "Settings" → "App-Level Environment Variables"
4. Add the following environment variable:
   - **Name**: `CLAUDE_API_KEY`
   - **Value**: Your actual Claude API key (starts with `sk-ant-api03-`)
   - **Type**: `SECRET` (encrypted)

### For Droplets (VPS):

```bash
# Set environment variable permanently
echo 'export CLAUDE_API_KEY="your-claude-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Or set it for the current session
export CLAUDE_API_KEY="your-claude-api-key-here"
```

### Verification Commands:

```bash
# Check if environment variable is set
echo $CLAUDE_API_KEY

# Run deployment check script
python deploy_check.py

# Test the application
curl http://localhost:8080/
```

## Docker Deployment:

```bash
# Build the image
docker build -t coherent-multiplex .

# Run with environment variable
docker run -p 8080:8080 -e CLAUDE_API_KEY="your-api-key" coherent-multiplex

# Or use docker-compose with .env file
docker-compose up -d
```

## Troubleshooting:

1. **"Claude API client not initialized"**
   - Check that CLAUDE_API_KEY environment variable is set
   - Verify the API key is valid and active

2. **"Authentication failed"**
   - Double-check your API key
   - Make sure there are no extra spaces or characters

3. **Connection errors**
   - Check internet connectivity
   - Verify firewall settings allow outbound HTTPS

## Security Notes:

- Never commit API keys to Git
- Use DigitalOcean's secret management features
- The Dockerfile automatically creates a secure config.py using environment variables
- Local config files are excluded via .dockerignore
