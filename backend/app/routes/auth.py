from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
from pydantic import BaseModel, EmailStr
import logging
from app.services.db import get_db

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = "mindguard_secret_key_2024"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    created_at: str
    last_login: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Verify JWT token and return token data."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        token_data = TokenData(email=email)
        return token_data
    except jwt.PyJWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current authenticated user."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    db = await get_db()
    users_collection = db["users"]
    user = await users_collection.find_one({"email": token_data.email})
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Authentication routes
@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        
    Returns:
        JWT token and user information
    """
    try:
        db = await get_db()
        users_collection = db["users"]
        
        # Check if user already exists
        existing_user = await users_collection.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate password strength
        if len(user_data.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Create user
        user_count = await users_collection.count_documents({})
        user_id = f"user_{user_count + 1}"
        hashed_password = get_password_hash(user_data.password)
        
        user = {
            "id": user_id,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "age": user_data.age,
            "gender": user_data.gender,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        # Save user to database
        await users_collection.insert_one(user)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data.email}, expires_delta=access_token_expires
        )
        
        logger.info(f"New user registered: {user_data.email}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                full_name=user["full_name"],
                age=user["age"],
                gender=user["gender"],
                created_at=user["created_at"],
                last_login=user["last_login"]
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during user registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """
    Authenticate user and return JWT token.
    
    Args:
        user_credentials: User login credentials
        
    Returns:
        JWT token and user information
    """
    try:
        db = await get_db()
        users_collection = db["users"]
        
        # Find user
        user = await users_collection.find_one({"email": user_credentials.email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not verify_password(user_credentials.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if user is active
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_credentials.email}, expires_delta=access_token_expires
        )
        
        # Update last login
        await users_collection.update_one(
            {"email": user_credentials.email},
            {"$set": {"last_login": datetime.utcnow().isoformat()}}
        )
        
        logger.info(f"User logged in: {user_credentials.email}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                full_name=user["full_name"],
                age=user["age"],
                gender=user["gender"],
                created_at=user["created_at"],
                last_login=datetime.utcnow().isoformat()
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during user login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )

@router.post("/logout")
async def logout(current_user: Dict = Depends(get_current_user)):
    """
    Logout user (invalidate token on client side).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    try:
        # In a production environment, you might want to add the token to a blacklist
        # For now, we'll just return a success message
        # The client should remove the token from storage
        
        logger.info(f"User logged out: {current_user['email']}")
        
        return {
            "message": "Successfully logged out",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user information
    """
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        age=current_user["age"],
        gender=current_user["gender"],
        created_at=current_user["created_at"],
        last_login=current_user["last_login"]
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Issue a new access token using the provided bearer token (treat as refresh token if you implement separate one)."""
    try:
        db = await get_db()
        users_collection = db["users"]
        
        incoming_token = credentials.credentials
        payload = jwt.decode(incoming_token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        
        user = await users_collection.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        
        new_token = create_access_token({"sub": email})
        return Token(
            access_token=new_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                full_name=user["full_name"],
                age=user["age"],
                gender=user["gender"],
                created_at=user["created_at"],
                last_login=user["last_login"]
            )
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

@router.put("/profile", response_model=UserResponse)
async def update_profile(
    profile_data: dict,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update user profile information.
    
    Args:
        profile_data: Profile data to update
        current_user: Current authenticated user
        
    Returns:
        Updated user information
    """
    try:
        # Update allowed fields
        allowed_fields = ["full_name", "age", "gender"]
        
        for field in allowed_fields:
            if field in profile_data:
                current_user[field] = profile_data[field]
        
        # Update user in database
        db = await get_db()
        users_collection = db["users"]
        await users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {field: current_user[field] for field in allowed_fields if field in profile_data}}
        )
        
        logger.info(f"Profile updated for user: {current_user['email']}")
        
        return UserResponse(
            id=current_user["id"],
            email=current_user["email"],
            full_name=current_user["full_name"],
            age=current_user["age"],
            gender=current_user["gender"],
            created_at=current_user["created_at"],
            last_login=current_user["last_login"]
        )
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during profile update"
        )

@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Change user password.
    
    Args:
        current_password: Current password
        new_password: New password
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    try:
        # Verify current password
        if not verify_password(current_password, current_user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Hash new password
        new_hashed_password = get_password_hash(new_password)
        current_user["hashed_password"] = new_hashed_password
        
        # Update user in database
        db = await get_db()
        users_collection = db["users"]
        await users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {"hashed_password": new_hashed_password}}
        )
        
        logger.info(f"Password changed for user: {current_user['email']}")
        
        return {
            "message": "Password changed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password change"
        )

@router.delete("/account")
async def delete_account(
    password: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete user account.
    
    Args:
        password: User password for confirmation
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    try:
        # Verify password
        if not verify_password(password, current_user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is incorrect"
            )
        
        # Remove user from database
        db = await get_db()
        users_collection = db["users"]
        await users_collection.delete_one({"email": current_user["email"]})
        
        logger.info(f"Account deleted for user: {current_user['email']}")
        
        return {
            "message": "Account deleted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during account deletion"
        )


