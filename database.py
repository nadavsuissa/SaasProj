from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, UTC
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Connection using environment variables
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017/constructionapp")
client = MongoClient(MONGO_CONNECTION_STRING)
db = client.constructionapp

# Collections
users = db.users
projects = db.projects
project_invitations = db.project_invitations  # New collection for invitations
tasks = db.tasks  # New collection for tasks

# New global storage for thread activity tracking
thread_activity = {}  # Maps thread_id -> last_activity_timestamp

# Add indexes for better performance
try:
    # Create indexes for common queries
    projects.create_index([("owner", 1)])
    projects.create_index([("members", 1)])
    projects.create_index([("_id", 1), ("owner", 1)])
except Exception as e:
    print(f"Error creating indexes: {e}")

def get_user_projects(username):
    """Get all projects owned by or accessible to the user"""
    try:
        user_projects = list(projects.find({
            "$or": [
                {"owner": username},
                {"members": username}
            ]
        }).sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for project in user_projects:
            project["_id"] = str(project["_id"])
        
        return user_projects
    except Exception as e:
        print(f"Error fetching user projects: {e}")
        return []

def get_project(project_id):
    """Get a project by ID"""
    try:
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        project = projects.find_one({"_id": project_obj_id})
        return project
    except Exception as e:
        print(f"Error getting project: {e}")
        return None

def create_project(name, description, owner, project_type):
    """Create a new project"""
    try:
        # Create project document
        project = {
            "name": name,
            "description": description,
            "owner": owner,
            "type": project_type,
            "status": "planning",  # Default status is planning (בתכנון)
            "members": [],
            "chat_history": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "assistant_id": None,
            "files": [],
            "thread_id": None
        }
        
        # Insert project
        result = projects.insert_one(project)
        
        # Return the inserted ID as string
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating project: {e}")
        return None

def update_project(project_id, update_data):
    """Update a project with new data"""
    try:
        update_data["updated_at"] = datetime.utcnow()
        projects.update_one(
            {"_id": ObjectId(project_id)},
            {"$set": update_data}
        )
        return True
    except:
        return False

def update_project_assistant(project_id, assistant_id):
    """Update a project with an OpenAI assistant ID"""
    try:
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        
        # Update project
        projects.update_one(
            {"_id": project_obj_id},
            {
                "$set": {
                    "assistant_id": assistant_id,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return True
    except Exception as e:
        print(f"Error updating project assistant: {e}")
        return False

def update_project_thread(project_id, thread_id):
    """Update a project with an OpenAI thread ID"""
    try:
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        
        # Update project
        projects.update_one(
            {"_id": project_obj_id},
            {
                "$set": {
                    "thread_id": thread_id,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return True
    except Exception as e:
        print(f"Error updating project thread: {e}")
        return False

def add_file_to_project(project_id, file_id, file_name, file_type):
    """Add a file ID to a project"""
    try:
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        
        file_info = {
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "uploaded_at": datetime.utcnow()
        }
        
        # Update project
        projects.update_one(
            {"_id": project_obj_id},
            {
                "$push": {"files": file_info},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return True
    except Exception as e:
        print(f"Error adding file to project: {e}")
        return False

def remove_file_from_project(project_id, file_id):
    """Remove a file ID from a project"""
    try:
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        
        # Update project
        projects.update_one(
            {"_id": project_obj_id},
            {
                "$pull": {"files": {"file_id": file_id}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return True
    except Exception as e:
        print(f"Error removing file from project: {e}")
        return False

def get_thread_last_activity(thread_id):
    """Get the timestamp of the last activity for a thread."""
    return thread_activity.get(thread_id)

def update_thread_activity(thread_id):
    """Update the last activity timestamp for a thread."""
    thread_activity[thread_id] = datetime.now(UTC)
    return thread_activity[thread_id]

def add_chat_message(project_id, sender, message, is_ai=False):
    """Add a chat message to a project."""
    # Find the project by ID
    project = projects.find_one({"_id": ObjectId(project_id)})
    
    if not project:
        print(f"Project {project_id} not found")
        return False
    
    # Get the current UTC time
    now = datetime.now(UTC)
    
    # Create a new message
    new_message = {
        "sender": sender,
        "message": message,
        "timestamp": now,
        "is_ai": is_ai
    }
    
    # Add the message to the project's chat_history
    projects.update_one(
        {"_id": ObjectId(project_id)},
        {"$push": {"chat_history": new_message}}
    )
    
    # Update thread activity timestamp if this is an AI message (indicates thread usage)
    if is_ai and project.get("thread_id"):
        update_thread_activity(project["thread_id"])
    
    print(f"Chat message from {sender} added to project {project_id}")
    return True

def add_project_member(project_id, username):
    """Add a member to a project"""
    try:
        projects.update_one(
            {"_id": ObjectId(project_id)},
            {
                "$addToSet": {"members": username},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return True
    except:
        return False

def remove_project_member(project_id, username):
    """Remove a member from a project"""
    try:
        projects.update_one(
            {"_id": ObjectId(project_id)},
            {
                "$pull": {"members": username},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        return True
    except:
        return False

def delete_project(project_id):
    """Delete a project"""
    try:
        projects.delete_one({"_id": ObjectId(project_id)})
        return True
    except:
        return False

def toggle_project_status(project_id):
    """Toggle project status between 'active' and 'pending'"""
    try:
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        
        # Get current project
        project = projects.find_one({"_id": project_obj_id})
        if not project:
            return False
        
        # Toggle status
        current_status = project.get("status", "active")
        new_status = "pending" if current_status == "active" else "active"
        
        # Update project
        projects.update_one(
            {"_id": project_obj_id},
            {
                "$set": {
                    "status": new_status,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return new_status
    except Exception as e:
        print(f"Error toggling project status: {e}")
        return False

def update_project_status(project_id, new_status):
    """Update project status to a specific status"""
    try:
        # Validate status
        valid_statuses = ["planning", "in_progress", "completed"]
        if new_status not in valid_statuses:
            print(f"Invalid status: {new_status}")
            return False
            
        # Convert string ID to ObjectId
        project_obj_id = ObjectId(project_id)
        
        # Update project
        projects.update_one(
            {"_id": project_obj_id},
            {
                "$set": {
                    "status": new_status,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return new_status
    except Exception as e:
        print(f"Error updating project status: {e}")
        return False

def update_user(user_id, update_data):
    """Update a user with new data"""
    try:
        # Convert string ID to ObjectId if it's a string
        user_obj_id = ObjectId(user_id) if isinstance(user_id, str) else user_id
        
        # Always update the updated_at timestamp
        if "updated_at" not in update_data:
            update_data["updated_at"] = datetime.utcnow()
        
        # Update user
        result = users.update_one(
            {"_id": user_obj_id},
            {"$set": update_data}
        )
        
        # Check if the update was successful
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating user: {e}")
        return False

def delete_user_account(user_id):
    """Delete a user account and all associated projects"""
    try:
        # Convert string ID to ObjectId if it's a string
        user_obj_id = ObjectId(user_id) if isinstance(user_id, str) else user_id
        
        # Find the user to get their username
        user = users.find_one({"_id": user_obj_id})
        if not user:
            print(f"User with ID {user_id} not found")
            return False
        
        username = user.get("username")
        
        # Delete all projects owned by this user
        projects_result = projects.delete_many({"owner": username})
        print(f"Deleted {projects_result.deleted_count} projects owned by {username}")
        
        # Delete the user
        user_result = users.delete_one({"_id": user_obj_id})
        
        # Check if the deletion was successful
        success = user_result.deleted_count > 0
        if success:
            print(f"Successfully deleted user {username}")
        else:
            print(f"Failed to delete user {username}")
        
        return success
    except Exception as e:
        print(f"Error deleting user account: {e}")
        return False

def send_project_invitation(project_id, sender_username, recipient_username):
    """Send a project invitation to a user"""
    try:
        # Check if project exists
        project = get_project(project_id)
        if not project:
            return False, "Project not found"
        
        # Check if sender is the owner of the project
        if project["owner"] != sender_username:
            return False, "Only the project owner can send invitations"
        
        # Check if recipient exists
        recipient = users.find_one({"username": recipient_username})
        if not recipient:
            return False, "Recipient user not found"
        
        # Check if recipient is already a member
        if recipient_username in project.get("members", []) or recipient_username == project["owner"]:
            return False, "User is already a member of this project"
        
        # Check if invitation already exists
        existing_invitation = project_invitations.find_one({
            "project_id": str(project["_id"]),
            "recipient": recipient_username,
            "status": "pending"
        })
        
        if existing_invitation:
            return False, "Invitation already sent to this user"
        
        # Create invitation
        invitation = {
            "project_id": str(project["_id"]),
            "project_name": project["name"],
            "sender": sender_username,
            "recipient": recipient_username,
            "status": "pending",
            "created_at": datetime.utcnow()
        }
        
        # Insert invitation
        project_invitations.insert_one(invitation)
        
        return True, "Invitation sent successfully"
    except Exception as e:
        print(f"Error sending project invitation: {e}")
        return False, f"Error: {str(e)}"

def get_user_invitations(username):
    """Get all pending invitations for a user"""
    try:
        invitations = list(project_invitations.find({
            "recipient": username,
            "status": "pending"
        }).sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for invitation in invitations:
            if "_id" in invitation:
                invitation["_id"] = str(invitation["_id"])
        
        return invitations
    except Exception as e:
        print(f"Error fetching user invitations: {e}")
        return []

def respond_to_invitation(invitation_id, username, accept):
    """Accept or reject a project invitation"""
    try:
        # Convert string ID to ObjectId
        invitation_obj_id = ObjectId(invitation_id)
        
        # Get invitation
        invitation = project_invitations.find_one({
            "_id": invitation_obj_id,
            "recipient": username
        })
        
        if not invitation:
            return False, "Invitation not found or not for this user"
        
        # Update invitation status
        status = "accepted" if accept else "rejected"
        project_invitations.update_one(
            {"_id": invitation_obj_id},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )
        
        # If accepted, add user to project members
        if accept:
            result = add_project_member(invitation["project_id"], username)
            if not result:
                return False, "Failed to add user to project"
        
        return True, f"Invitation {status} successfully"
    except Exception as e:
        print(f"Error responding to invitation: {e}")
        return False, f"Error: {str(e)}"

# Task Management Functions
def create_task(project_id, creator_username, assigned_to, title, description, due_date=None, priority="medium"):
    """Create a new task in a project"""
    try:
        # Create task document
        task = {
            "project_id": project_id,
            "creator": creator_username,
            "assigned_to": assigned_to,
            "title": title,
            "description": description,
            "status": "pending",  # pending, in_progress, completed
            "priority": priority,  # low, medium, high
            "due_date": due_date,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "completed_at": None,
            "is_read": False  # To track if the assigned user has read the task
        }
        
        # Insert task
        result = tasks.insert_one(task)
        
        # Return the inserted ID as string
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating task: {e}")
        return None

def get_project_tasks(project_id):
    """Get all tasks for a project"""
    try:
        project_tasks = list(tasks.find({"project_id": project_id}).sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for task in project_tasks:
            task["_id"] = str(task["_id"])
        
        return project_tasks
    except Exception as e:
        print(f"Error fetching project tasks: {e}")
        return []

def get_user_tasks(username):
    """Get all tasks assigned to a user"""
    try:
        user_tasks = list(tasks.find({"assigned_to": username}).sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for task in user_tasks:
            task["_id"] = str(task["_id"])
            
            # Add project name to each task
            try:
                project = projects.find_one({"_id": ObjectId(task["project_id"])})
                if project:
                    task["project_name"] = project["name"]
                else:
                    task["project_name"] = "Unknown Project"
            except:
                task["project_name"] = "Unknown Project"
        
        return user_tasks
    except Exception as e:
        print(f"Error fetching user tasks: {e}")
        return []

def get_task(task_id):
    """Get a task by ID"""
    try:
        # Convert string ID to ObjectId
        task_obj_id = ObjectId(task_id)
        task = tasks.find_one({"_id": task_obj_id})
        
        if task:
            task["_id"] = str(task["_id"])
            # Add project name to task
            try:
                project = projects.find_one({"_id": ObjectId(task["project_id"])})
                if project:
                    task["project_name"] = project["name"]
                else:
                    task["project_name"] = "Unknown Project"
            except:
                task["project_name"] = "Unknown Project"
        
        return task
    except Exception as e:
        print(f"Error getting task: {e}")
        return None

def update_task(task_id, update_data):
    """Update a task with new data"""
    try:
        update_data["updated_at"] = datetime.utcnow()
        
        # If status is being changed to completed, set completed_at
        if update_data.get("status") == "completed":
            update_data["completed_at"] = datetime.utcnow()
        
        tasks.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": update_data}
        )
        return True
    except Exception as e:
        print(f"Error updating task: {e}")
        return False

def mark_task_as_read(task_id):
    """Mark a task as read by the assigned user"""
    try:
        tasks.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {"is_read": True, "updated_at": datetime.utcnow()}}
        )
        return True
    except Exception as e:
        print(f"Error marking task as read: {e}")
        return False

def delete_task(task_id):
    """Delete a task"""
    try:
        tasks.delete_one({"_id": ObjectId(task_id)})
        return True
    except Exception as e:
        print(f"Error deleting task: {e}")
        return False

def get_unread_tasks_count(username):
    """Get the count of unread tasks for a user"""
    try:
        return tasks.count_documents({
            "assigned_to": username,
            "is_read": False
        })
    except Exception as e:
        print(f"Error getting unread tasks count: {e}")
        return 0 