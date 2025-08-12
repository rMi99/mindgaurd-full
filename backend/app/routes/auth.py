from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional
import jwt
import bcrypt
import uuid
from pydantic import BaseModel, EmailStr
from app.services.db import db

router = APIRouter()
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = "mindguard_secret_key_2024"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    username: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TempUserLink(BaseModel):
    temp_user_id: str
    email: EmailStr
    password: str
    username: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: str
    username: Optional[str] = None
    is_temporary: bool = False

class TempUserResponse(BaseModel):
    temp_user_id: str
    message: str

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@router.post("/auth/temp-user", response_model=TempUserResponse)
async def create_temp_user():
    """Create a temporary user for anonymous usage"""
    try:
        temp_user_id = f"temp_{uuid.uuid4().hex[:12]}"
        
        # Create temporary user record
        temp_user = {
            "temp_user_id": temp_user_id,
            "created_at": datetime.utcnow().isoformat(),
            "is_temporary": True,
            "linked_to_registered": False
        }
        
        await db.temp_users.insert_one(temp_user)
        
        return TempUserResponse(
            temp_user_id=temp_user_id,
            message="Temporary user created successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create temporary user: {str(e)}")

@router.post("/auth/register", response_model=AuthResponse)
async def register_user(user_data: UserRegistration):
    """Register a new user with email and password"""
    try:
        # Check if user already exists
        existing_user = await db.users.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Create user record
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_record = {
            "user_id": user_id,
            "email": user_data.email,
            "username": user_data.username or user_data.email.split('@')[0],
            "password_hash": hashed_password,
            "created_at": datetime.utcnow().isoformat(),
            "is_temporary": False,
            "auth_provider": "email"
        }
        
        await db.users.insert_one(user_record)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            email=user_data.email,
            username=user_record["username"],
            is_temporary=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/auth/login", response_model=AuthResponse)
async def login_user(user_data: UserLogin):
    """Login user with email and password"""
    try:
        # Find user by email
        user = await db.users.find_one({"email": user_data.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password
        if not verify_password(user_data.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["user_id"]}, expires_delta=access_token_expires
        )
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user["user_id"],
            email=user["email"],
            username=user.get("username"),
            is_temporary=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.post("/auth/link-temp-user", response_model=AuthResponse)
async def link_temp_user_to_account(link_data: TempUserLink):
    """Link temporary user data to a new registered account"""
    try:
        # Check if temp user exists
        temp_user = await db.temp_users.find_one({"temp_user_id": link_data.temp_user_id})
        if not temp_user:
            raise HTTPException(status_code=404, detail="Temporary user not found")
        
        if temp_user.get("linked_to_registered"):
            raise HTTPException(status_code=400, detail="Temporary user already linked")
        
        # Check if email already exists
        existing_user = await db.users.find_one({"email": link_data.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        hashed_password = hash_password(link_data.password)
        
        # Create new registered user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_record = {
            "user_id": user_id,
            "email": link_data.email,
            "username": link_data.username or link_data.email.split('@')[0],
            "password_hash": hashed_password,
            "created_at": datetime.utcnow().isoformat(),
            "is_temporary": False,
            "auth_provider": "email",
            "linked_from_temp": link_data.temp_user_id
        }
        
        await db.users.insert_one(user_record)
        
        # Transfer temporary user data to registered user
        await transfer_temp_user_data(link_data.temp_user_id, user_id)
        
        # Mark temp user as linked
        await db.temp_users.update_one(
            {"temp_user_id": link_data.temp_user_id},
            {"$set": {"linked_to_registered": True, "linked_user_id": user_id}}
        )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            email=link_data.email,
            username=user_record["username"],
            is_temporary=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link temporary user: {str(e)}")

async def transfer_temp_user_data(temp_user_id: str, registered_user_id: str):
    """Transfer data from temporary user to registered user"""
    try:
        # Transfer assessments
        await db.user_assessments.update_many(
            {"user_id": temp_user_id},
            {"$set": {"user_id": registered_user_id, "transferred_from_temp": temp_user_id}}
        )
        
        # Transfer any other user-specific data collections here
        # For example, if there are journal entries, mood logs, etc.
        
    except Exception as e:
        print(f"Error transferring temp user data: {str(e)}")
        # Don't raise exception here to avoid breaking the registration process

@router.get("/auth/verify-token")
async def verify_token(current_user: str = Depends(get_current_user)):
    """Verify if the current token is valid"""
    try:
        user = await db.users.find_one({"user_id": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "valid": True,
            "user_id": current_user,
            "email": user.get("email"),
            "username": user.get("username"),
            "is_temporary": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token verification failed: {str(e)}")

@router.post("/auth/logout")
async def logout_user():
    """Logout user (client-side token removal)"""
    return {"message": "Logged out successfully"}

