from fastapi import FastAPI, Request, Form, HTTPException, Depends, status, Cookie, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict, Any
from pathlib import Path
from database import (users, get_user_projects, get_project, create_project, update_project, 
                   add_project_member, remove_project_member, delete_project, 
                   toggle_project_status, update_project_status, update_user, delete_user_account,
                   send_project_invitation, get_user_invitations, respond_to_invitation,
                   create_task, get_project_tasks, get_user_tasks, get_task, update_task, mark_task_as_read, delete_task, get_unread_tasks_count,
                   update_project_assistant, update_project_thread, add_file_to_project, remove_file_from_project,
                   add_chat_message, db)
from bson.objectid import ObjectId
import os
import json
import ai_helper
import logging
from dotenv import load_dotenv
import time
import asyncio

# Helper function to get the current request
def get_request(request: Request):
    return request

# Helper function to check if a user is a member of a project
def is_project_member(project, username):
    if not project or not username:
        return False
    return username == project.get("owner") or username in project.get("members", [])

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

app = FastAPI(title="Project Management API")

# Create templates directory if it doesn't exist
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create uploads directory if it doesn't exist
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(32).hex())  # Generate random secret key if not provided
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Log warning if default secret key is used
if SECRET_KEY == os.urandom(32).hex():
    logger.warning("Using randomly generated SECRET_KEY. Sessions will be invalidated on restart.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.connection_count: Dict[str, int] = {}  # Track connection count per project

    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
            self.connection_count[project_id] = 0
        self.active_connections[project_id].append(websocket)
        self.connection_count[project_id] += 1
        logger.info(f"New WebSocket connection for project {project_id}. Total connections: {self.connection_count[project_id]}")

    def disconnect(self, websocket: WebSocket, project_id: str):
        if project_id in self.active_connections:
            if websocket in self.active_connections[project_id]:
                self.active_connections[project_id].remove(websocket)
                self.connection_count[project_id] -= 1
                logger.info(f"WebSocket disconnected for project {project_id}. Remaining connections: {self.connection_count[project_id]}")
            if self.connection_count[project_id] <= 0:
                # Clean up if no more connections for this project
                del self.active_connections[project_id]
                del self.connection_count[project_id]
                logger.info(f"Removed project {project_id} from connection manager (no active connections)")

    async def send_message(self, message: str, project_id: str):
        if project_id in self.active_connections:
            disconnected = []
            for i, connection in enumerate(self.active_connections[project_id]):
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket {i} for project {project_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up any connections that failed
            for conn in disconnected:
                if conn in self.active_connections[project_id]:
                    self.active_connections[project_id].remove(conn)
                    self.connection_count[project_id] -= 1
                    logger.info(f"Removed dead connection for project {project_id}. Remaining: {self.connection_count[project_id]}")

manager = ConnectionManager()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_current_user(access_token: Optional[str] = Cookie(None, alias="access_token")):
    if not access_token:
        return None
    try:
        if access_token.startswith("Bearer "):
            token = access_token.replace("Bearer ", "")
        else:
            token = access_token
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    
    user = users.find_one({"username": username})
    if user is None:
        return None
    return user

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: dict = Depends(get_current_user)):
    # Add current_user to the context immediately
    context = {"request": request, "title": "לוח בקרה", "current_user": current_user, "user": current_user}

    if current_user:
        # Fetch user projects
        user_projects = get_user_projects(current_user["username"])
        context["projects"] = user_projects
        context["projects_count"] = len(user_projects)
        
        # Get user's tasks (both created by them and assigned to them)
        user_tasks = get_user_tasks(current_user["username"])
        context["tasks"] = user_tasks
        context["tasks_count"] = len(user_tasks)
        
        # Count documents (files) across all projects
        documents_count = 0
        for project in user_projects:
            if "files" in project:
                documents_count += len(project["files"])
        context["documents_count"] = documents_count
        
    return templates.TemplateResponse("index.html", context)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "title": "Login"
        }
    )

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = users.find_one({"username": username})
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "title": "Login",
                "error": "Invalid username or password"
            }
        )
    
    # Check both possible password field names for backward compatibility
    if "password" in user:
        password_field = "password"
    elif "hashed_password" in user:
        password_field = "hashed_password"
    else:
        print(f"User document has no password field: {user.keys()}")
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "title": "Login",
                "error": "Account configuration error. Please contact support."
            }
        )
    
    if not verify_password(password, user[password_field]):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "title": "Login",
                "error": "Invalid username or password"
            }
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username},
        expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "title": "Register"
        }
    )

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    email: Optional[str] = Form(None)  # Make email optional
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "title": "Register",
                "error": "Passwords do not match"
            }
        )
    
    if users.find_one({"username": username}):
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "title": "Register",
                "error": "Username already exists"
            }
        )
    
    # Create user
    user = {
        "username": username,
        "password": get_password_hash(password),
        "created_at": datetime.utcnow()
    }
    
    # Add email if provided
    if email:
        user["email"] = email
        
    users.insert_one(user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username},
        expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

@app.get("/myprojects", response_class=HTMLResponse)
async def my_projects(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get user's projects
    projects = get_user_projects(current_user["username"])
    
    return templates.TemplateResponse(
        "myprojects.html",
        {
            "request": request,
            "title": "My Projects",
            "user": current_user,
            "projects": projects
        }
    )

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    try:
        # Fix date fields in user object
        user_date_fields = ["created_at", "updated_at"]
        for field in user_date_fields:
            if field in current_user and isinstance(current_user[field], str):
                try:
                    current_user[field] = datetime.fromisoformat(current_user[field].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    current_user[field] = datetime.utcnow()
        
        # Get user's projects
        projects = get_user_projects(current_user["username"])
        
        # Fix date fields in projects
        for project in projects:
            project_date_fields = ["created_at", "updated_at"]
            for field in project_date_fields:
                if field in project and isinstance(project[field], str):
                    try:
                        project[field] = datetime.fromisoformat(project[field].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        project[field] = datetime.utcnow()
        
        # Get user's pending invitations
        invitations = get_user_invitations(current_user["username"])
        
        # Fix date fields in invitations
        for invitation in invitations:
            if "created_at" in invitation and isinstance(invitation["created_at"], str):
                try:
                    invitation["created_at"] = datetime.fromisoformat(invitation["created_at"].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    invitation["created_at"] = datetime.utcnow()
        
        # Get user's tasks
        user_tasks = get_user_tasks(current_user["username"])
        
        # Fix date fields in tasks
        for task in user_tasks:
            task_date_fields = ["created_at", "updated_at", "completed_at", "due_date"]
            for field in task_date_fields:
                if field in task and task[field] is not None and isinstance(task[field], str):
                    try:
                        task[field] = datetime.fromisoformat(task[field].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        if field != "due_date":  # Keep due_date as None if invalid
                            task[field] = datetime.utcnow()
        
        # Get unread tasks count
        unread_tasks = get_unread_tasks_count(current_user["username"])
        
        # Add current time for date comparisons in the template
        now = datetime.now()
        
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "title": "הפרופיל שלי",
                "user": current_user,
                "projects": projects,
                "invitations": invitations,
                "tasks": user_tasks,
                "unread_tasks": unread_tasks,
                "now": now  # Pass current datetime
            }
        )
    except Exception as e:
        print(f"Error in profile: {e}")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error_message": f"An error occurred while viewing your profile: {str(e)}",
                "user": current_user
            }
        )

@app.post("/update_profile", response_class=HTMLResponse)
async def update_profile(
    request: Request, 
    email: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Update user with new information
    update_data = {"updated_at": datetime.utcnow()}
    
    if email:
        update_data["email"] = email
    
    # Add the update_user function to database.py if it doesn't exist
    from database import update_user
    success = update_user(str(current_user["_id"]), update_data)
    
    # Get updated user data
    updated_user = users.find_one({"_id": ObjectId(current_user["_id"])})
    
    # Get user's projects for the statistics
    projects = get_user_projects(current_user["username"])
    
    if success:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "title": "הפרופיל שלי",
                "user": updated_user,
                "projects": projects,
                "success": "הפרופיל עודכן בהצלחה"
            }
        )
    else:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "title": "הפרופיל שלי",
                "user": current_user,
                "projects": projects,
                "error": "אירעה שגיאה בעדכון הפרופיל"
            }
        )
        
@app.post("/delete_account", response_class=HTMLResponse)
async def delete_account(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # This is a serious operation, so we should add confirmation in the UI
    from database import delete_user_account
    success = delete_user_account(str(current_user["_id"]))
    
    if success:
        # Clear the session cookie
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.delete_cookie(key="access_token")
        return response
    else:
        # Get user's projects for the statistics
        projects = get_user_projects(current_user["username"])
        
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "title": "הפרופיל שלי",
                "user": current_user,
                "projects": projects,
                "error": "אירעה שגיאה במחיקת החשבון"
            }
        )

@app.get("/projects/new", response_class=HTMLResponse)
async def new_project_page(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse(
        "create_project.html",
        {
            "request": request,
            "title": "Create Project",
            "user": current_user
        }
    )

@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_detail(request: Request, project_id: str, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    logger.info(f"Entering project_detail for project_id: {project_id}")
    try:
        # Get project details
        logger.info(f"Attempting to fetch project with ID: {project_id}")
        project = get_project(project_id)
        logger.info(f"Project data fetched: {type(project)}")
        
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "title": "Project Not Found",
                    "error_message": "The project you're looking for doesn't exist or has been deleted.",
                    "user": current_user
                },
                status_code=404
            )
        
        # Check if user has access to this project
        if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
            logger.warning(f"Access denied for user {current_user['username']} to project {project_id}")
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "title": "Access Denied",
                    "error_message": "You don't have permission to access this project.",
                    "user": current_user
                },
                status_code=403
            )
        
        logger.info(f"Access check passed for project: {project.get('name', 'N/A')}")
        
        # Convert ObjectId to string for JSON serialization
        project["_id"] = str(project["_id"])
        
        # Fix date fields - convert strings to datetime objects if needed
        date_fields = ["created_at", "updated_at"]
        for field in date_fields:
            if field in project and isinstance(project[field], str):
                try:
                    # Try to parse the date string (if it's ISO format)
                    project[field] = datetime.fromisoformat(project[field].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    # If parsing fails, create a dummy datetime to prevent template errors
                    project[field] = datetime.utcnow()
                    logger.warning(f"Converted invalid date string in {field} to current datetime")
        
        logger.info("Project date fields processed")
        
        # Get project tasks
        project_tasks = get_project_tasks(project_id)
        logger.info(f"Tasks fetched: {len(project_tasks)} tasks")
        
        # Fix date fields in tasks
        for task in project_tasks:
            task_date_fields = ["created_at", "updated_at", "completed_at", "due_date"]
            for field in task_date_fields:
                if field in task and task[field] is not None and isinstance(task[field], str):
                    try:
                        task[field] = datetime.fromisoformat(task[field].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        if field != "due_date":  # Keep due_date as None if invalid
                            task[field] = datetime.utcnow()
        
        logger.info("Task date fields processed")
        
        # Prepare sorted & sliced recent tasks for the overview
        try:
            # Sort tasks by 'updated_at' descending, handling potential missing keys/None values
            recent_tasks = sorted(
                project_tasks,
                key=lambda task: task.get('updated_at', task.get('created_at', datetime.min)),
                reverse=True
            )[:5]
        except Exception as sort_err:
            logger.error(f"Error sorting tasks for project {project_id}: {sort_err}", exc_info=True)
            recent_tasks = [] # Default to empty list on error
            
        # Add to context
        context = {
            "request": request,
            "title": project["name"],
            "project": project,
            "user": current_user,
            "is_owner": current_user["username"] == project.get("owner"),
            "project_tasks": project_tasks, # Keep original list for other parts
            "recent_tasks": recent_tasks, # Add the sorted/sliced list
            "members": project.get("members", []),
            "now": datetime.now()
        }
        
        # Get member user objects for the Members tab
        member_usernames = project.get("members", [])
        member_users = []
        for username in member_usernames:
            user = users.find_one({"username": username})
            if user:
                # Remove sensitive fields
                if "password" in user:
                    del user["password"]
                if "hashed_password" in user:
                    del user["hashed_password"]
                member_users.append(user)
        
        # Update context with member user objects
        context["members"] = member_users
        
        logger.info("Context prepared, attempting to render template")
        
        return templates.TemplateResponse("project_detail.html", context)
    except Exception as e:
        logger.error(f"Error in project_detail for project_id {project_id}: {e}", exc_info=True)
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error_message": f"An error occurred while retrieving the project: {str(e)}",
                "user": current_user
            },
            status_code=500
        )

@app.post("/projects/{project_id}/upload")
async def upload_project_file(
    project_id: str, 
    file: UploadFile = File(...), 
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Get project details
        logger.info(f"File upload request for project {project_id}, file: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
        project = get_project(project_id)
        
        if not project:
            logger.warning(f"Project not found for file upload: {project_id}")
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if user has access to this project
        if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
            logger.warning(f"Access denied for user {current_user['username']} to project {project_id} during file upload")
            raise HTTPException(status_code=403, detail="You don't have access to this project")
        
        # Check file size - OpenAI has a 512MB limit
        MAX_FILE_SIZE = 512 * 1024 * 1024  # 512MB in bytes
        file_contents = await file.read()
        file_size = len(file_contents)
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE} bytes)")
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size is 512MB. Your file is {file_size / (1024 * 1024):.2f}MB")
        
        logger.info(f"File read successfully: {file.filename}, size: {file_size} bytes")
        
        # Check file type - make sure it's a supported format
        valid_extensions = ['.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.jpeg', '.jpg', '.png']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.warning(f"Unsupported file type: {file_ext}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported types are: {', '.join(valid_extensions)}"
            )
        
        # Upload file to OpenAI
        logger.info(f"Uploading file to OpenAI: {file.filename}")
        file_id = await ai_helper.upload_file(file_contents, file.filename)
        
        if not file_id:
            logger.warning("Failed to upload file to OpenAI (no file_id returned)")
            raise HTTPException(status_code=500, detail="Failed to upload file to OpenAI")
        
        logger.info(f"File uploaded to OpenAI successfully: {file_id}")
        
        # Get assistant ID
        assistant_id = project.get("assistant_id")
        
        # Create assistant if it doesn't exist
        if not assistant_id:
            logger.info(f"Creating new assistant for project: {project['name']}")
            assistant_id = await ai_helper.create_project_assistant(
                project["name"], 
                project["type"], 
                project["description"]
            )
            
            if not assistant_id:
                logger.warning("Failed to create assistant")
                await ai_helper.delete_file(file_id)
                raise HTTPException(status_code=500, detail="Failed to create assistant")
            
            # Update project with assistant ID
            update_project_assistant(project_id, assistant_id)
            project["assistant_id"] = assistant_id
            logger.info(f"Assistant created: {assistant_id}")
        
        # Create thread if it doesn't exist
        thread_id = project.get("thread_id")
        if not thread_id:
            logger.info(f"Creating new thread for project: {project['name']}")
            thread_id = await ai_helper.create_project_thread()
            if not thread_id:
                logger.warning("Failed to create thread")
                await ai_helper.delete_file(file_id)
                raise HTTPException(status_code=500, detail="Failed to create thread")
            
            # Update project with thread ID
            update_project_thread(project_id, thread_id)
            project["thread_id"] = thread_id
            logger.info(f"Thread created: {thread_id}")
        
        # Attach file to assistant
        logger.info(f"Attaching file {file_id} to assistant {assistant_id}")
        result = await ai_helper.attach_file_to_assistant(assistant_id, file_id)
        
        if not result:
            # Clean up by deleting the file if it couldn't be attached
            logger.warning(f"Failed to attach file {file_id} to assistant {assistant_id}")
            await ai_helper.delete_file(file_id)
            raise HTTPException(status_code=500, detail="Failed to attach file to assistant")
        
        logger.info(f"File attached to assistant successfully")
        
        # Add file to project in database
        add_file_to_project(project_id, file_id, file.filename, file.content_type)
        logger.info(f"File {file_id} added to project {project_id} in database")
        
        # Save file locally as well
        file_path = os.path.join(uploads_dir, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(file_contents)
        logger.info(f"File saved locally: {file_path}")
        
        return {"success": True, "file_id": file_id, "file_name": file.filename}
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        error_msg = f"Error uploading file: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.delete("/projects/{project_id}/files/{file_id}")
async def delete_project_file(
    project_id: str,
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get project details
    project = get_project(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if user has access to this project
    if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
        raise HTTPException(status_code=403, detail="You don't have access to this project")
    
    try:
        # Get assistant ID
        assistant_id = project.get("assistant_id")
        
        if not assistant_id:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # Detach file from assistant
        await ai_helper.detach_file_from_assistant(assistant_id, file_id)
        
        # Delete file from OpenAI
        await ai_helper.delete_file(file_id)
        
        # Remove file from project in database
        remove_file_from_project(project_id, file_id)
        
        # Delete local file if it exists
        for filename in os.listdir(uploads_dir):
            if filename.startswith(f"{file_id}_"):
                os.remove(os.path.join(uploads_dir, filename))
        
        return {"success": True}
    
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.post("/projects/create")
async def create_new_project(
    request: Request,
    name: str = Form(...),
    project_type: str = Form(...),
    description: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    try:
        # Create new project
        project_id = create_project(
            name=name,
            description=description,
            owner=current_user["username"],
            project_type=project_type
        )
        
        # Create assistant and thread when the project is created
        logger.info(f"Creating assistant and thread for new project {project_id}")
        
        # Create assistant
        assistant_id = await ai_helper.create_project_assistant(name, project_type, description)
        if assistant_id:
            logger.info(f"Created assistant {assistant_id} for project {project_id}")
            update_project_assistant(project_id, assistant_id)
            
            # Create thread
            thread_id = await ai_helper.create_project_thread()
            if thread_id:
                logger.info(f"Created thread {thread_id} for project {project_id}")
                update_project_thread(project_id, thread_id)
        
        return RedirectResponse(url=f"/projects/{project_id}", status_code=status.HTTP_302_FOUND)
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        # Redirect to an error page or back to the form with an error message
        return templates.TemplateResponse(
            "create_project.html",
            {
                "request": request,
                "title": "Create Project",
                "user": current_user,
                "error": f"Error creating project: {str(e)}"
            }
        )

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time chat with project assistant."""
    logger.info(f"WebSocket connection attempt for project {project_id}")
    
    # Prepare for potential cleanup
    connection_accepted = False
    
    try:
        # Get the token from query parameters or cookies
        token = websocket.query_params.get("token")
        if not token:
            cookie_header = websocket.headers.get("cookie", "")
            for cookie in cookie_header.split(";"):
                if cookie.strip().startswith("access_token="):
                    token = cookie.strip().split("=")[1]
                    break
        
        if not token:
            logger.warning("Authentication failed: No token provided")
            await websocket.close(code=1008, reason="Not authenticated")
            return
        
        # Verify the token and get the user
        try:
            if token.startswith("Bearer "):
                token = token.replace("Bearer ", "")
            
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                logger.warning("Authentication failed: Invalid token payload")
                await websocket.close(code=1008, reason="Invalid token")
                return
            
            logger.info(f"Authenticated user: {username}")
            
            user = users.find_one({"username": username})
            if not user:
                logger.warning(f"User not found: {username}")
                await websocket.close(code=1008, reason="User not found")
                return
            
            # Check if user has access to the project
            project = get_project(project_id)
            if not project:
                logger.warning(f"Project not found: {project_id}")
                await websocket.close(code=1008, reason="Project not found")
                return
            
            if username not in project.get("members", []) and username != project.get("owner"):
                logger.warning(f"Access denied for user {username} to project {project_id}")
                await websocket.close(code=1008, reason="Not authorized to access this project")
                return
            
            logger.info(f"WebSocket connection accepted for user {username} in project {project_id}")
            
            # Connect to the websocket
            await manager.connect(websocket, project_id)
            connection_accepted = True
            
            # Send connection confirmation message to user
            welcome_message = {
                "type": "system",
                "message": "Connected to chat server",
                "timestamp": datetime.now(UTC).isoformat()
            }
            await websocket.send_text(json.dumps(welcome_message))
            
            # Main message processing loop
            async for data in websocket.iter_text():
                try:
                    logger.info(f"Message received from user {username}: {data[:50]}...")
                    
                    message_data = json.loads(data)
                    
                    # Handle history request
                    if message_data.get("action") == "get_history":
                        logger.info(f"Chat history requested for project {project_id}")
                        project = get_project(project_id)
                        if project and "chat_history" in project:
                            # Convert datetime objects to strings
                            history = []
                            for msg in project["chat_history"]:
                                msg_copy = msg.copy()
                                # Convert timestamp to string if it's a datetime
                                if "timestamp" in msg_copy and isinstance(msg_copy["timestamp"], datetime):
                                    msg_copy["timestamp"] = msg_copy["timestamp"].isoformat()
                                history.append(msg_copy)
                            
                            # Send back the chat history
                            history_response = {
                                "action": "history_response",
                                "messages": history
                            }
                            await websocket.send_text(json.dumps(history_response))
                            logger.info(f"Sent {len(project['chat_history'])} chat history messages")
                        else:
                            # Send empty history
                            await websocket.send_text(json.dumps({
                                "action": "history_response", 
                                "messages": []
                            }))
                            logger.info("No chat history found or project not found")
                        continue
                    
                    user_message = message_data.get("message", "")
                    
                    if not user_message.strip():
                        logger.warning(f"Empty message received from user {username}")
                        continue
                    
                    # Save user message to database
                    add_chat_message(project_id, username, user_message)
                    logger.info(f"User message saved to database")
                    
                    # Format and send the user message to all connected clients
                    user_message_obj = {
                        "sender": username,
                        "message": user_message,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "is_ai": False
                    }
                    await manager.send_message(json.dumps(user_message_obj), project_id)
                    logger.info(f"User message sent to all clients")
                    
                    # Get AI response using the Assistants API
                    logger.info(f"Getting AI response for message: {user_message[:50]}...")
                    
                    try:
                        # Send a "typing" indicator to clients
                        typing_message = {
                            "sender": "AI Assistant",
                            "message": "מעבד",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "is_ai": True,
                            "is_typing": True
                        }
                        await manager.send_message(json.dumps(typing_message), project_id)
                        
                        # Get AI response using the Assistants API with streaming
                        full_response = ""
                        has_response = False
                        streaming_started = False
                        
                        # Set a timeout for AI response (5 minutes)
                        start_time = time.time()
                        max_wait_time = 300  # 5 minutes in seconds
                        
                        try:
                            async for chunk in ai_helper.get_ai_response_stream(project, user_message):
                                # Check for timeout
                                if time.time() - start_time > max_wait_time:
                                    logger.warning(f"AI response timeout after {max_wait_time} seconds")
                                    break
                                    
                                if chunk:
                                    # Skip chunks that indicate no response yet
                                    if "לא התקבלה תשובה מהעוזר" in chunk:
                                        continue
                                        
                                    has_response = True
                                    
                                    # Ensure each chunk is treated as a discrete unit
                                    if not streaming_started:
                                        streaming_started = True
                                        logger.info("Starting streaming response.")
                                    
                                    full_response += chunk
                                    # Send each chunk to the client
                                    chunk_message = {
                                        "sender": "AI Assistant",
                                        "message": chunk,
                                        "timestamp": datetime.now(UTC).isoformat(),
                                        "is_ai": True,
                                        "is_streaming": True
                                    }
                                    await manager.send_message(json.dumps(chunk_message), project_id)
                                    # Add a small delay to ensure proper message sequence
                                    await asyncio.sleep(0.05)
                        except Exception as e:
                            logger.error(f"Error during AI response streaming: {e}", exc_info=True)
                            if not has_response:
                                raise Exception(f"Failed to get AI response: {str(e)}")
                        
                        # Save complete AI response to database
                        if full_response:
                            add_chat_message(project_id, "AI Assistant", full_response, is_ai=True)
                            logger.info(f"AI response saved to database (length: {len(full_response)})")
                            
                            # Send completion message if we were streaming
                            if streaming_started:
                                complete_message = {
                                    "sender": "AI Assistant",
                                    "message": "",  # Empty as we already sent the content in chunks
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "is_ai": True,
                                    "stream_complete": True
                                }
                                await manager.send_message(json.dumps(complete_message), project_id)
                            # If we never sent any chunks, send the complete response now
                            elif not streaming_started:
                                complete_message = {
                                    "sender": "AI Assistant",
                                    "message": full_response,
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "is_ai": True
                                }
                                await manager.send_message(json.dumps(complete_message), project_id)
                        else:
                            # If we didn't get a response but didn't raise an exception
                            if not has_response:
                                logger.warning("No response received from AI but no exception was raised")
                                raise Exception("No response received from AI")
                            
                    except Exception as e:
                        logger.error(f"Error getting AI response: {e}", exc_info=True)
                        error_message = "שגיאה בעיבוד הבקשה. אנא נסה שוב."
                        error_message_obj = {
                            "sender": "AI Assistant",
                            "message": error_message,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "is_ai": True,
                            "is_error": True
                        }
                        await manager.send_message(json.dumps(error_message_obj), project_id)
                        # Save error message to database
                        add_chat_message(project_id, "AI Assistant", error_message, is_ai=True)
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid message format",
                        "timestamp": datetime.now(UTC).isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Error processing your message",
                        "timestamp": datetime.now(UTC).isoformat()
                    }))
                
        except JWTError as e:
            logger.error(f"JWT Error: {e}")
            await websocket.close(code=1008, reason=f"Invalid token: {str(e)}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for project {project_id}")
        if connection_accepted:
            manager.disconnect(websocket, project_id)
    except Exception as e:
        logger.error(f"Unexpected error in websocket endpoint: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=f"Server error")
        except Exception:
            pass  # Already closed
    finally:
        # Ensure connection is properly removed
        if connection_accepted:
            try:
                manager.disconnect(websocket, project_id)
            except Exception as e:
                logger.error(f"Error during WebSocket cleanup: {e}")

@app.get("/api/projects/{project_id}/overview")
async def get_project_overview_api(
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Get project details
        project = get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if user has access to this project
        if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
            raise HTTPException(status_code=403, detail="You don't have access to this project")
        
        # Get overview data from the assistant
        overview_data = await ai_helper.get_project_overview(project)
        
        # If there's an error in the overview data, return it properly
        if "error" in overview_data:
            return {"error": overview_data["error"]}
            
        # Return the overview data as JSON
        return overview_data
        
    except Exception as e:
        print(f"Error getting project overview: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting project overview: {str(e)}")

@app.get("/api/projects/{project_id}/graphs")
async def get_project_graphs_api(
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Get project details
        project = get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if user has access to this project
        if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
            raise HTTPException(status_code=403, detail="You don't have access to this project")
        
        # Get graph data from the assistant
        graph_data = await ai_helper.get_project_graph_data(project)
        
        # If there's an error in the graph data, return it properly
        if "error" in graph_data:
            logger.error(f"Error fetching graph data from AI: {graph_data['error']}")
            return {"error": graph_data["error"]}
            
        # Return the graph data as JSON (now contains the 'visualizations' structure)
        return graph_data
        
    except Exception as e:
        # Use structured logging for error reporting
        logger.error(f"Error getting project graph data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting project graph data: {str(e)}")

@app.post("/api/projects/{project_id}/update-status")
async def update_project_status_api(
    project_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Get the new status from request body
        body = await request.json()
        new_status = body.get("status")
        
        if not new_status:
            raise HTTPException(status_code=400, detail="Status not provided")
            
        # Validate status
        valid_statuses = ["planning", "in_progress", "completed"]
        if new_status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status: {new_status}. Valid statuses are: {', '.join(valid_statuses)}")
        
        # Get project details
        project = get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check if user has access to this project
        if current_user["username"] != project.get("owner"):
            raise HTTPException(status_code=403, detail="Only the project owner can change the status")
        
        # Update project status
        result = update_project_status(project_id, new_status)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to update project status")
            
        # Return the new status
        return {"status": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating project status: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating project status: {str(e)}")

# Project invitation endpoints
@app.post("/projects/{project_id}/invite")
async def invite_to_project(
    project_id: str,
    request: Request,
    username: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Send invitation
    success, message = send_project_invitation(project_id, current_user["username"], username)
    
    # Redirect back to project page with success/error message
    if success:
        return RedirectResponse(
            url=f"/projects/{project_id}?success={message}#project-members", 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return RedirectResponse(
            url=f"/projects/{project_id}?error={message}#project-members", 
            status_code=status.HTTP_302_FOUND
        )

@app.post("/invitations/{invitation_id}/accept")
async def accept_invitation(
    invitation_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Accept invitation
    success, message = respond_to_invitation(invitation_id, current_user["username"], True)
    
    # Redirect back to profile page with success/error message
    if success:
        return RedirectResponse(
            url=f"/profile?success={message}", 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return RedirectResponse(
            url=f"/profile?error={message}", 
            status_code=status.HTTP_302_FOUND
        )

@app.post("/invitations/{invitation_id}/reject")
async def reject_invitation(
    invitation_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Reject invitation
    success, message = respond_to_invitation(invitation_id, current_user["username"], False)
    
    # Redirect back to profile page with success/error message
    if success:
        return RedirectResponse(
            url=f"/profile?success={message}", 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return RedirectResponse(
            url=f"/profile?error={message}", 
            status_code=status.HTTP_302_FOUND
        )

@app.get("/projects/{project_id}/members", response_class=HTMLResponse)
async def project_members(
    project_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get project details
    project = get_project(project_id)
    
    if not project:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Project not found",
                "user": current_user
            }
        )
    
    # Check if user has access to this project
    if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "You don't have access to this project",
                "user": current_user
            }
        )
    
    # Get member user objects
    member_usernames = project.get("members", [])
    member_users = []
    for username in member_usernames:
        user = users.find_one({"username": username})
        if user:
            # Remove sensitive fields
            if "password" in user:
                del user["password"]
            if "hashed_password" in user:
                del user["hashed_password"]
            member_users.append(user)
    
    return templates.TemplateResponse(
        "project_members.html",
        {
            "request": request,
            "title": f"{project['name']} - Members",
            "user": current_user,
            "project": project,
            "members": member_users,
            "is_owner": current_user["username"] == project.get("owner")
        }
    )

@app.post("/projects/{project_id}/remove_member/{username}")
async def remove_member(
    project_id: str,
    username: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get project details
    project = get_project(project_id)
    
    if not project:
        return RedirectResponse(
            url=f"/myprojects?error=Project not found", 
            status_code=status.HTTP_302_FOUND
        )
    
    # Check if user is the owner
    if current_user["username"] != project.get("owner"):
        return RedirectResponse(
            url=f"/projects/{project_id}?error=Only the project owner can remove members", 
            status_code=status.HTTP_302_FOUND
        )
    
    # Remove member
    result = remove_project_member(project_id, username)
    
    if result:
        return RedirectResponse(
            url=f"/projects/{project_id}?success=Member+removed+successfully#project-members", 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return RedirectResponse(
            url=f"/projects/{project_id}?error=Failed+to+remove+member#project-members", 
            status_code=status.HTTP_302_FOUND
        )

@app.get("/api/users/search")
async def search_users(username_prefix: str, current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Search for users whose username starts with the given prefix
        # Exclude the current user from the results
        found_users = list(users.find(
            {
                "$and": [
                    {"username": {"$regex": f"^{username_prefix}", "$options": "i"}},
                    {"username": {"$ne": current_user["username"]}}
                ]
            },
            {"username": 1, "_id": 0}  # Only return the username field
        ).limit(10))  # Limit to 10 results
        
        return {"users": [user["username"] for user in found_users]}
    except Exception as e:
        print(f"Error searching users: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching users: {str(e)}")

@app.get("/api/users/unread-notifications")
async def get_unread_notifications_count(current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Count pending invitations
        from database import project_invitations
        invitation_count = project_invitations.count_documents({
            "recipient": current_user["username"],
            "status": "pending"
        })
        
        return {"count": invitation_count}
    except Exception as e:
        print(f"Error counting notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Error counting notifications: {str(e)}")

# Task Management Routes
@app.post("/projects/{project_id}/tasks", response_class=HTMLResponse)
async def create_project_task(
    project_id: str,
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
    assigned_to: str = Form(...),
    priority: str = Form("medium"),
    due_date: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get project details
    project = get_project(project_id)
    
    if not project:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Project not found",
                "user": current_user
            }
        )
    
    # Check if user has access to this project
    if current_user["username"] not in project.get("members", []) and current_user["username"] != project.get("owner"):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "You don't have access to this project",
                "user": current_user
            }
        )
    
    # Check if assigned user is part of the project
    if assigned_to != project.get("owner") and assigned_to not in project.get("members", []):
        return RedirectResponse(
            url=f"/projects/{project_id}?error=User+not+in+project", 
            status_code=status.HTTP_302_FOUND
        )
    
    # Convert date string to datetime if provided
    parsed_due_date = None
    if due_date:
        try:
            parsed_due_date = datetime.strptime(due_date, "%Y-%m-%d")
        except:
            parsed_due_date = None
    
    # Create task
    task_id = create_task(
        project_id=project_id,
        creator_username=current_user["username"],
        assigned_to=assigned_to,
        title=title,
        description=description,
        due_date=parsed_due_date,
        priority=priority
    )
    
    if task_id:
        return RedirectResponse(
            url=f"/projects/{project_id}?success=Task+created+successfully", 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return RedirectResponse(
            url=f"/projects/{project_id}?error=Failed+to+create+task", 
            status_code=status.HTTP_302_FOUND
        )

@app.get("/tasks/{task_id}")
async def view_task(
    task_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    # Check if user is logged in first
    if not current_user:
        return RedirectResponse(url="/login?next=/tasks/" + task_id, status_code=302)
        
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    project_id = task.get("project_id")
    if not project_id:
        raise HTTPException(status_code=404, detail="Task has no associated project")
        
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    if not is_project_member(project, current_user["username"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    # Mark task as read by this user
    if current_user["username"] != task.get("creator", project.get("owner")):
        task["is_read"] = True
        db.tasks.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {"is_read": True}}
        )

    now = datetime.now()
    return templates.TemplateResponse(
        "task_detail.html",
        {
            "request": request,
            "title": task["title"],
            "user": current_user,
            "task": task,
            "project": project,
            "now": now,
            "can_edit": current_user["username"] == task.get("creator", project.get("owner")) or current_user["username"] == project.get("owner")
        }
    )

@app.post("/tasks/{task_id}/update", response_class=HTMLResponse)
async def update_task_status(
    task_id: str,
    request: Request,
    task_status: str = Form(..., alias="status"),
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get task details
    task = get_task(task_id)
    
    if not task:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Task not found",
                "user": current_user
            }
        )
    
    # Check if user is assigned to this task or is the creator or project owner
    project = get_project(task["project_id"])
    if not project:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Project not found",
                "user": current_user
            }
        )
    
    if (current_user["username"] != task["assigned_to"] and
        current_user["username"] != task["creator"] and
        current_user["username"] != project.get("owner")):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "You don't have permission to update this task",
                "user": current_user
            }
        )
    
    # Update task status
    success = update_task(task_id, {"status": task_status})
    
    if success:
        # Redirect based on where the update came from
        redirect_url = request.headers.get("referer", f"/tasks/{task_id}")
        return RedirectResponse(
            url=redirect_url, 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Failed to update task status",
                "user": current_user
            }
        )

@app.post("/tasks/{task_id}/delete", response_class=HTMLResponse)
async def delete_task_route(
    task_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get task details
    task = get_task(task_id)
    
    if not task:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Task not found",
                "user": current_user
            }
        )
    
    # Check if user is the creator or project owner
    project = get_project(task["project_id"])
    if not project:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Project not found",
                "user": current_user
            }
        )
    
    if current_user["username"] != task["creator"] and current_user["username"] != project.get("owner"):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "You don't have permission to delete this task",
                "user": current_user
            }
        )
    
    # Delete task
    success = delete_task(task_id)
    
    if success:
        # Redirect based on where the delete came from
        redirect_url = request.headers.get("referer", f"/projects/{task['project_id']}")
        return RedirectResponse(
            url=redirect_url, 
            status_code=status.HTTP_302_FOUND
        )
    else:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": "Failed to delete task",
                "user": current_user
            }
        )

# API endpoint to get unread tasks count for notification badge
@app.get("/api/unread-tasks-count", response_class=JSONResponse)
async def get_unread_tasks_count_api(current_user: dict = Depends(get_current_user)):
    if not current_user:
        return {"count": 0}
    
    count = get_unread_tasks_count(current_user["username"])
    return {"count": count}

@app.post("/tasks/{task_id}/comments")
async def add_task_comment(
    request: Request,
    task_id: str,
    type: str = Form(...),
    content: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    # Check if user is logged in first
    if not current_user:
        return RedirectResponse(url="/login?next=/tasks/" + task_id, status_code=302)
        
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    project = get_project(task["project_id"])
    if not is_project_member(project, current_user["username"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    comment = {
        "author": current_user["username"],
        "type": type,
        "content": content,
        "created_at": datetime.now()
    }
    
    # Update the task with the new comment
    db.tasks.update_one(
        {"_id": ObjectId(task_id)},
        {"$push": {"comments": comment}}
    )
    
    # Also add to chat if it's a status update
    if type == "status":
        add_chat_message(task["project_id"], current_user["username"], 
                        f"עדכון סטטוס למשימה '{task['title']}': {content}", is_ai=False)
    
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)

@app.delete("/projects/{project_id}/delete")
async def delete_project_endpoint(
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    logger.info(f"Attempting to delete project {project_id} by user {current_user['username']}")
    
    try:
        project = get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for deletion.")
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Authorization: Only owner can delete
        if project.get("owner") != current_user["username"]:
            logger.warning(f"User {current_user['username']} is not the owner of project {project_id}. Deletion denied.")
            raise HTTPException(status_code=403, detail="Only the project owner can delete the project")
        
        logger.info(f"Authorization successful for deleting project {project_id}")
        
        assistant_id = project.get("assistant_id")
        thread_id = project.get("thread_id")
        project_files = project.get("files", [])
        
        # 1. Delete/Detach Files from OpenAI and Local Storage
        if assistant_id and project_files:
            logger.info(f"Processing {len(project_files)} files for deletion from assistant {assistant_id}")
            for file_info in project_files:
                file_id = file_info.get("file_id")
                file_name = file_info.get("file_name", "unknown_file")
                if file_id:
                    logger.info(f"Detaching file {file_id} ({file_name}) from assistant {assistant_id}")
                    detach_success = await ai_helper.detach_file_from_assistant(assistant_id, file_id)
                    if not detach_success:
                        logger.warning(f"Failed to detach file {file_id} from assistant {assistant_id}. Continuing cleanup.")
                    
                    logger.info(f"Deleting file {file_id} ({file_name}) from OpenAI")
                    delete_openai_success = await ai_helper.delete_file(file_id)
                    if not delete_openai_success:
                        logger.warning(f"Failed to delete file {file_id} from OpenAI. Continuing cleanup.")
                else:
                    logger.warning(f"Skipping file detachment/deletion due to missing file_id for file: {file_name}")

                # Delete local file
                try:
                    local_file_deleted = False
                    for filename in os.listdir(uploads_dir):
                        # Match based on file_id prefix or full name if ID is missing
                        match_pattern = f"{file_id}_" if file_id else file_name
                        if filename.startswith(match_pattern):
                            local_file_path = os.path.join(uploads_dir, filename)
                            os.remove(local_file_path)
                            logger.info(f"Deleted local file: {local_file_path}")
                            local_file_deleted = True
                            break # Assume one local file per entry
                    if not local_file_deleted:
                         logger.warning(f"Local file not found or already deleted for {file_name} (ID: {file_id})")
                except Exception as local_delete_err:
                    logger.error(f"Error deleting local file for {file_name} (ID: {file_id}): {local_delete_err}")

        # 2. Delete OpenAI Assistant
        if assistant_id:
            logger.info(f"Deleting OpenAI assistant {assistant_id}")
            delete_assistant_success = await ai_helper.delete_openai_assistant(assistant_id)
            if not delete_assistant_success:
                logger.warning(f"Failed to delete OpenAI assistant {assistant_id}. Continuing cleanup.")
        
        # 3. Delete OpenAI Thread
        if thread_id:
            logger.info(f"Deleting OpenAI thread {thread_id}")
            delete_thread_success = await ai_helper.delete_openai_thread(thread_id)
            if not delete_thread_success:
                logger.warning(f"Failed to delete OpenAI thread {thread_id}. Continuing cleanup.")

        # 4. Delete Project Tasks from Database
        try:
            logger.info(f"Deleting tasks associated with project {project_id}")
            task_deletion_result = db.tasks.delete_many({"project_id": project_id})
            logger.info(f"Deleted {task_deletion_result.deleted_count} tasks for project {project_id}.")
        except Exception as task_delete_err:
             logger.error(f"Error deleting tasks for project {project_id}: {task_delete_err}")
             # Decide if this is critical - perhaps log and continue?

        # 5. Delete Project Invitations from Database
        try:
             logger.info(f"Deleting invitations associated with project {project_id}")
             invitation_deletion_result = db.project_invitations.delete_many({"project_id": project_id})
             logger.info(f"Deleted {invitation_deletion_result.deleted_count} invitations for project {project_id}.")
        except Exception as inv_delete_err:
             logger.error(f"Error deleting invitations for project {project_id}: {inv_delete_err}")

        # 6. Delete Project from Database
        logger.info(f"Deleting project document {project_id} from database.")
        delete_db_success = delete_project(project_id) # Assumes database.py has delete_project
        
        if not delete_db_success:
            # This is more critical. Maybe raise an internal server error?
            logger.error(f"CRITICAL: Failed to delete project {project_id} from database after cleaning other resources.")
            raise HTTPException(status_code=500, detail="Failed to delete project from database. Please contact support.")

        logger.info(f"Project {project_id} deleted successfully by user {current_user['username']}.")
        return JSONResponse(content={"success": True, "message": "Project deleted successfully"}, status_code=200)

    except HTTPException as e:
        # Re-raise HTTP exceptions (like 403, 404)
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during project deletion for {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during project deletion: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} (reload: {reload})")
    uvicorn.run(app, host=host, port=port, reload=reload) 
