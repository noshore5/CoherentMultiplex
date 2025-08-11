#!/usr/bin/env python3
"""
Deployment Version Checker for DigitalOcean
This script helps identify if DO is using your latest code
"""

import os
import time
import json
from datetime import datetime

def create_version_info():
    """Create a version info file that gets deployed with your app"""
    version_info = {
        "build_time": datetime.now().isoformat(),
        "timestamp": int(time.time()),
        "git_commit": "latest",  # You can add git hash here if needed
        "environment": "production",
        "claude_utils_version": "v2.0_fixed",
        "deployment_check": True
    }
    
    with open("version_info.json", "w") as f:
        json.dump(version_info, f, indent=2)
    
    print(f"✅ Created version_info.json with timestamp: {version_info['build_time']}")
    return version_info

def check_deployment_requirements():
    """Check that all required files exist for deployment"""
    required_files = [
        "main.py",
        "requirements.txt", 
        "Dockerfile",
        "utils/claude_utils.py",
        "config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def verify_claude_utils():
    """Verify claude_utils.py has no syntax errors"""
    try:
        with open("utils/claude_utils.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for common issues
        if "unde" in content and content.count("unde") == 1 and "under" not in content:
            print("❌ Found incomplete 'unde' text in claude_utils.py")
            return False
            
        if content.count('"""') % 2 != 0:
            print("❌ Unmatched triple quotes in claude_utils.py")
            return False
            
        print("✅ claude_utils.py appears syntactically correct")
        return True
        
    except UnicodeDecodeError as e:
        print(f"❌ Encoding error in claude_utils.py: {e}")
        print("   This will cause 'undefined' errors on DigitalOcean!")
        return False
    except Exception as e:
        print(f"❌ Error checking claude_utils.py: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DigitalOcean Deployment Verification")
    print("=" * 50)
    
    # Create version info
    version_info = create_version_info()
    
    # Check requirements
    requirements_ok = check_deployment_requirements()
    
    # Verify claude_utils
    claude_ok = verify_claude_utils()
    
    print("\n" + "=" * 50)
    
    if requirements_ok and claude_ok:
        print("✅ Ready for deployment!")
        print(f"📝 Version timestamp: {version_info['timestamp']}")
        print("\nTo force DigitalOcean to use latest code:")
        print("1. Commit all changes: git add . && git commit -m 'force rebuild'")
        print("2. Push to main: git push origin main")
        print("3. In DO dashboard: Settings → Force Rebuild")
        print("4. Or destroy and recreate the app")
    else:
        print("❌ Issues found - fix before deploying")
        
    print("\nAfter deployment, check: https://your-app.ondigitalocean.app/version_info.json")
