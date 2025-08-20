#!/usr/bin/env python3
"""
MindGuard Backend Setup Script
This script helps set up the MindGuard backend environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "logs",
        "data",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_env_file():
    """Create .env file with configuration."""
    env_content = """# MindGuard Backend Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Security
SECRET_KEY=mindguard_secret_key_2024
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database (for future use)
DATABASE_URL=sqlite:///./mindguard.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/mindguard.log

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Model Configuration
MODEL_PATH=models/health_model.pkl
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️  .env file already exists")

def create_sample_data():
    """Create sample data for testing."""
    sample_data = {
        "users": [
            {
                "email": "test@mindguard.com",
                "password": "testpass123",
                "full_name": "Test User",
                "age": 30,
                "gender": "male"
            }
        ],
        "assessments": []
    }
    
    # This would typically be stored in a database
    print("✅ Sample data structure created")

def main():
    """Main setup function."""
    print("🚀 MindGuard Backend Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create environment file
    print("\n⚙️  Creating configuration...")
    create_env_file()
    
    # Create sample data
    print("\n📊 Creating sample data...")
    create_sample_data()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate your virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    print("2. Start the server: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("3. Open your browser to: http://localhost:8000/docs")
    print("\nHappy coding! 🚀")

if __name__ == "__main__":
    main() 