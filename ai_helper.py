import os
import asyncio
import tempfile
import re
import logging
import time
from datetime import datetime, timedelta
from functools import partial
from openai import OpenAI
import json
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_helper")

# Set OpenAI API key from environment variable with no default
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY environment variable not set!")
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants for timeouts and retries
API_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Wrapper for API calls with error handling and logging
async def api_call(func, func_name, **kwargs):
    """Wrapper for API calls with error handling and logging"""
    start_time = time.time()
    logger.info(f"API Call: {func_name} - Args: {kwargs}")
    
    for attempt in range(MAX_RETRIES):
        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(**kwargs)),
                timeout=API_TIMEOUT
            )
            
            elapsed = time.time() - start_time
            logger.info(f"API Response: {func_name} - Success - Time: {elapsed:.2f}s")
            return response
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"API TIMEOUT ERROR: {func_name} - Attempt {attempt+1}/{MAX_RETRIES} - Time: {elapsed:.2f}s")
            if attempt == MAX_RETRIES - 1:
                raise
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            
            if "rate_limit" in error_msg.lower():
                logger.error(f"API RATE LIMIT ERROR: {func_name} - {error_msg} - Time: {elapsed:.2f}s")
                # Don't retry rate limit errors
                raise
            elif "timeout" in error_msg.lower():
                logger.error(f"API TIMEOUT ERROR: {func_name} - {error_msg} - Time: {elapsed:.2f}s")
            else:
                logger.error(f"API ERROR: {func_name} - {error_msg} - Time: {elapsed:.2f}s")
            
            # For other errors, retry with exponential backoff if not the last attempt
            if attempt < MAX_RETRIES - 1:
                backoff_time = RETRY_DELAY * (2 ** attempt)
                logger.info(f"Retrying in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
            else:
                # Re-raise the exception if this was the last attempt
                raise

async def create_project_assistant(project_name: str, project_type: str, project_description: str) -> str:
    """
    Create a new assistant for a project with enhanced instructions.
    The instructions now mention that the assistant should adapt to different document types.
    """
    try:
        logger.info(f"Creating assistant for project: {project_name}")
        instructions = (
            f"אתה עוזר בנייה מקצועי לפרויקט בתחום {project_type}.\n"
            f"תיאור הפרויקט: {project_description}\n\n"
            "בעת עיבוד מסמכים, עליך לקרוא את הנתונים ולהשתמש בהם במדויק. "
            "ואם מדובר בטקסט חופשי או בשילוב נתונים, טפל בהם בצורה דינמית ומתאימה. "
            "השתמש במסמכים שהועלו כמקור מידע ראשוני והתאם את עיבוד הנתונים לסוג המסמך.\n\n"
            "עלייך לענות בעברית בלבד, בנוסף אתה תמיד מחוייב לענות ברמה הכי מפורטת שאתה מכול על בסיס הבנת על המידע שהעלתי לך."
        )
        
        # Create vector store first
        vector_store = await api_call(
            client.vector_stores.create,
            "vector_stores.create",
            name=f"Vector Store for {project_name}"
        )
        logger.info(f"Created vector store: {vector_store.id}")
        
        # Create assistant with both tools properly configured
        assistant = await api_call(
            client.beta.assistants.create,
            "assistants.create",
            name=f"פרויקט: {project_name}",
            instructions=instructions,
            model="gpt-4o",
            temperature=0.2,
            tools=[
                {"type": "file_search"}
            ],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store.id]
                }
            }
        )
        logger.info(f"Created assistant {assistant.id} with vector store {vector_store.id}")
        return assistant.id
    except Exception as e:
        logger.error(f"Error creating assistant: {e}")
        return None


async def create_project_thread() -> str:
    """Create a new thread for a project."""
    try:
        logger.info("Creating new thread")
        thread = await api_call(
            client.beta.threads.create,
            "threads.create"
        )
        logger.info(f"Created thread: {thread.id}")
        return thread.id
    except Exception as e:
        logger.error(f"Error creating thread: {e}")
        return None


async def upload_file(file_content: bytes, file_name: str) -> str:
    """Upload a file to OpenAI with improved file handling and resource management."""
    tmp_path = None
    try:
        # Create a unique filename based on timestamp and random bytes
        _, file_extension = os.path.splitext(file_name)
        tmp_dir = tempfile.gettempdir()
        tmp_filename = f"openai_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}{file_extension}"
        tmp_path = os.path.join(tmp_dir, tmp_filename)
        logger.info(f"Uploading file: {file_name}, extension: {file_extension}, temp path: {tmp_path}")
        
        # Write file to disk with proper resource management
        with open(tmp_path, 'wb') as f:
            f.write(file_content)
            f.flush()
            os.fsync(f.fileno())
        
        # Verify file was written successfully
        if not os.path.exists(tmp_path):
            logger.error(f"Failed to create temporary file: {tmp_path}")
            return None
            
        file_size = os.path.getsize(tmp_path)
        logger.info(f"Temporary file created successfully: {tmp_path}, size: {file_size} bytes")
        
        # Upload file to OpenAI
        loop = asyncio.get_running_loop()
        
        # Use a function that creates a new file handle each time to avoid file handle issues
        async def upload_with_timeout():
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.files.create(
                    file=open(tmp_path, 'rb'),
                    purpose="assistants"
                )),
                timeout=API_TIMEOUT
            )
            
        try:
            file = await upload_with_timeout()
            logger.info(f"File uploaded successfully with ID: {file.id}")
            return file.id
        except Exception as upload_error:
            logger.error(f"Error during OpenAI file upload: {upload_error}")
            # Re-raise to be handled by caller
            raise
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        return None
    finally:
        # Clean up the temporary file in all cases
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Temporary file removed: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove temporary file: {cleanup_error}")
                # On Windows, try to schedule file for deletion on next reboot if we can't delete it now
                if os.name == 'nt':
                    try:
                        import ctypes
                        MOVEFILE_DELAY_UNTIL_REBOOT = 0x4
                        ctypes.windll.kernel32.MoveFileExW(tmp_path, None, MOVEFILE_DELAY_UNTIL_REBOOT)
                        logger.info("Scheduled file for deletion on next reboot")
                    except Exception as inner_e:
                        logger.warning(f"Failed to schedule deletion: {inner_e}")


async def attach_file_to_assistant(assistant_id: str, file_id: str) -> bool:
    try:
        # First get assistant to extract vector store ID
        loop = asyncio.get_running_loop()
        def get_assistant():
            return client.beta.assistants.retrieve(assistant_id=assistant_id)
        assistant = await loop.run_in_executor(None, get_assistant)
        
        # Extract vector store ID(s) from assistant
        vector_store_id = None
        tool_resources = getattr(assistant, 'tool_resources', None)
        
        if tool_resources and hasattr(tool_resources, 'file_search') and tool_resources.file_search:
            if hasattr(tool_resources.file_search, 'vector_store_ids') and tool_resources.file_search.vector_store_ids:
                vector_store_id = tool_resources.file_search.vector_store_ids[0]
                print(f"Found vector store ID: {vector_store_id}")
        
        if not vector_store_id:
            print("No vector store found, creating one")
            def create_vector_store():
                return client.vector_stores.create(name=f"Vector Store for Assistant {assistant_id}")
            vector_store = await loop.run_in_executor(None, create_vector_store)
            vector_store_id = vector_store.id
            
            # Update assistant with the new vector store
            def update_assistant():
                return client.beta.assistants.update(
                    assistant_id=assistant_id,
                    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
                )
            await loop.run_in_executor(None, update_assistant)
            print(f"Created and attached vector store {vector_store_id}")
        
        # Add file to vector store
        def add_file():
            return client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)
        
        file_addition = await loop.run_in_executor(None, add_file)
        print(f"File {file_id} added to vector store {vector_store_id}")
        
        # Check file processing status
        max_polls = 20
        delay = 1
        for i in range(max_polls):
            def check_file():
                return client.vector_stores.files.retrieve(vector_store_id=vector_store_id, file_id=file_id)
            file_status = await loop.run_in_executor(None, check_file)
            
            if file_status.status == "completed":
                print(f"File processing completed for {file_id}")
                return True
            
            if file_status.status == "failed":
                print(f"File processing failed for {file_id}: {getattr(file_status, 'error', 'Unknown error')}")
                return False
            
            print(f"File processing in progress: {file_status.status}, waiting...")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5)
        
        print(f"File processing timed out after {max_polls} attempts")
        return False
    except Exception as e:
        print(f"Error attaching file: {e}")
        import traceback
        traceback.print_exc()
        return False


async def delete_file(file_id: str) -> bool:
    """Delete a file from OpenAI."""
    try:
        loop = asyncio.get_running_loop()
        delete_file_func = partial(client.files.delete, file_id=file_id)
        await loop.run_in_executor(None, delete_file_func)
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False


async def detach_file_from_assistant(assistant_id: str, file_id: str) -> bool:
    """Detach a file from an assistant's vector store."""
    try:
        loop = asyncio.get_running_loop()
        def get_assistant():
            return client.beta.assistants.retrieve(assistant_id=assistant_id)
        assistant = await loop.run_in_executor(None, get_assistant)
        
        vector_store_id = None
        # Handle tool_resources as an object with attributes rather than a dictionary
        tool_resources = getattr(assistant, 'tool_resources', None)
        
        if tool_resources:
            # Print details for debugging
            print(f"Tool resources type: {type(tool_resources)}")
            
            # Try different approaches to extract the vector_store_ids
            try:
                # Approach 1: Try to access it as a nested attribute
                if hasattr(tool_resources, 'file_search'):
                    file_search = tool_resources.file_search
                    if hasattr(file_search, 'vector_store_ids') and file_search.vector_store_ids:
                        vector_store_id = file_search.vector_store_ids[0]
                        print(f"Found vector store ID from attributes: {vector_store_id}")
            except Exception as attr_error:
                print(f"Error accessing tool_resources attributes: {attr_error}")
                
                # Approach 2: Try to convert to dictionary
                try:
                    tool_resources_dict = vars(tool_resources)
                    if 'file_search' in tool_resources_dict:
                        file_search = tool_resources_dict['file_search']
                        if isinstance(file_search, dict) and 'vector_store_ids' in file_search:
                            vector_store_ids = file_search['vector_store_ids']
                            if vector_store_ids:
                                vector_store_id = vector_store_ids[0]
                                print(f"Found vector store ID from dictionary: {vector_store_id}")
                except Exception as dict_error:
                    print(f"Error converting tool_resources to dictionary: {dict_error}")
                    
                # Approach 3: Try to access it as JSON
                try:
                    tool_resources_json = tool_resources.model_dump() if hasattr(tool_resources, 'model_dump') else None
                    if tool_resources_json and 'file_search' in tool_resources_json:
                        file_search = tool_resources_json['file_search']
                        if isinstance(file_search, dict) and 'vector_store_ids' in file_search:
                            vector_store_ids = file_search['vector_store_ids']
                            if vector_store_ids:
                                vector_store_id = vector_store_ids[0]
                                print(f"Found vector store ID from JSON: {vector_store_id}")
                except Exception as json_error:
                    print(f"Error converting tool_resources to JSON: {json_error}")
        
        if not vector_store_id:
            print(f"No vector store found for assistant {assistant_id}")
            return True
        
        def delete_file_from_vector_store():
            return client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)
        try:
            await loop.run_in_executor(None, delete_file_from_vector_store)
            print(f"File {file_id} removed from vector store {vector_store_id}")
            return True
        except Exception as delete_error:
            print(f"Error removing file from vector store: {delete_error}")
            if "file not found" in str(delete_error).lower():
                return True
            return False
    except Exception as e:
        print(f"Error detaching file from assistant: {e}")
        import traceback
        traceback.print_exc()
        return False


async def add_message_to_thread(thread_id: str, message: str, user: str) -> str:
    """Add a user message to a thread."""
    try:
        logger.info(f"Adding message to thread {thread_id} (first 30 chars): {message[:30]}...")
        message_obj = await api_call(
            client.beta.threads.messages.create,
            "threads.messages.create",
            thread_id=thread_id,
            role="user",
            content=message
        )
        logger.info(f"Added message: {message_obj.id}")
        return message_obj.id
    except Exception as e:
        logger.error(f"Error adding message to thread: {e}")
        return None


async def run_assistant(thread_id: str, assistant_id: str) -> str:
    """Run the assistant on a thread."""
    try:
        logger.info(f"Running assistant {assistant_id} on thread {thread_id}")
        run = await api_call(
            client.beta.threads.runs.create,
            "threads.runs.create",
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logger.info(f"Started run: {run.id}")
        return run.id
    except Exception as e:
        logger.error(f"Error running assistant: {e}")
        return None


async def get_run_status(thread_id: str, run_id: str) -> str:
    """Get the status of an assistant run."""
    try:
        run = await api_call(
            client.beta.threads.runs.retrieve,
            "threads.runs.retrieve",
            thread_id=thread_id,
            run_id=run_id
        )
        logger.info(f"Run {run_id} status: {run.status}")
        return run.status
    except Exception as e:
        logger.error(f"Error getting run status: {e}")
        return "failed"


async def get_run_result(thread_id: str, ignore_empty=False) -> str:
    """
    Fetch the latest assistant response from a thread.
    Use dynamic processing of the Hebrew text to decide whether to apply table parsing or simple cleaning.
    
    Args:
        thread_id: The ID of the thread to fetch messages from
        ignore_empty: If True, return empty string instead of "No response" message when no messages are found
    """
    try:
        logger.info(f"Getting run result from thread {thread_id}")
        messages = await api_call(
            client.beta.threads.messages.list,
            "threads.messages.list",
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        
        if messages.data and messages.data[0].role == "assistant":
            raw_text_parts = []
            for content_item in messages.data[0].content:
                if content_item.type == "text":
                    raw_text_parts.append(content_item.text.value)
            raw_text = "\n".join(raw_text_parts).strip()
            logger.info(f"Retrieved assistant response (first 50 chars): {raw_text[:50]}...")
            return raw_text
        
        if messages.data:
            logger.warning(f"Latest message is not from assistant. Role: {messages.data[0].role}")
        else:
            logger.warning(f"No messages found in thread {thread_id}")
            
        return "" if ignore_empty else "לא התקבלה תשובה מהעוזר."
    except Exception as e:
        logger.error(f"Error getting run result: {e}")
        return f"Error: {str(e)}"


def clean_citations(text):
    if not text:
        return text

    print(f"RAW CHARACTER DEBUG: {repr(text)}")
    
    # Extract JSON code blocks before processing farewell messages
    json_blocks = []
    json_block_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    
    # Find all JSON code blocks and store them for later reinsertion
    for i, match in enumerate(re.finditer(json_block_pattern, text, re.DOTALL)):
        placeholder = f"__JSON_BLOCK_{i}__"
        json_blocks.append((placeholder, match.group(0)))
        text = text.replace(match.group(0), placeholder)
    
    # Extract farewell message first, before any other processing
    farewell = ""
    farewell_phrases = [
        'אשמח לענות על כל שאלה נוספת',
        'אשמח לענות על שאלות נוספות',
        'בכבוד רב, עוזר הפרויקט',
        'בכבוד רב,',
        'לשירותכם,',
        'בברכה,'
    ]
    
    # Find the earliest farewell phrase in the text
    earliest_pos = -1
    matched_phrase = ""
    
    for phrase in farewell_phrases:
        pos = text.find(phrase)
        if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
            earliest_pos = pos
            matched_phrase = phrase
    
    # If we found a farewell phrase, extract everything from that point
    if earliest_pos != -1:
        farewell = text[earliest_pos:]
        text = text[:earliest_pos].strip()
        print(f"EXTRACTED FAREWELL: {farewell}")
        
        # Clean up the remaining text
        if text.endswith('.'):
            pass  # Already has a period
        elif text.endswith(','):
            text = text[:-1] + '.'  # Replace comma with period
        else:
            text += '.'  # Add period if missing
    
    # Flag to indicate if we've extracted a farewell
    has_extracted_farewell = bool(farewell)
    
    # 1) Remove ALL source citation markers like【20:0†openai_upload_20250327145107_5a53ef1b.docx】
    pattern_source_citation = r'【\d+:\d+†[^】]*】'
    if re.search(pattern_source_citation, text):
        print("Removing source citation markers")
        text = re.sub(pattern_source_citation, '', text)

    # 2) Directly remove the glyph-wrapped citation: citeturn0file0
    pattern_wrapped_citation = r'citeturn\d*file\d*'
    if re.search(pattern_wrapped_citation, text):
        print("Directly removing glyph-wrapped citation markers.")
        text = re.sub(pattern_wrapped_citation, '', text)
    
    # 3) If that leaves behind extra spaces or punctuation, clean it up
    text = re.sub(r'\s+\.', '.', text).strip()
    if text and not text.endswith('.'):
        text += '.'

    # 4) Run existing logic for standard citations
    text = standard_citation_cleanup(text)

    # 5) Add paragraph breaks where appropriate
    text = format_message_with_paragraphs(text)
    
    # 6) Enhance message with respectful tone for CEO audience
    text = enhance_message_tone(text, has_extracted_farewell)
    
    # 7) Re-insert JSON blocks before adding back farewell
    for placeholder, json_block in json_blocks:
        text = text.replace(placeholder, json_block)
    
    # 8) Add back the farewell message at the very end, after all other processing
    if farewell:
        text += f"\n\n{farewell}"

    return text

def standard_citation_cleanup(text):
    """Clean up standard citation markers from text."""
    # Regular processing if specific patterns not found
    if "citeturn" in text:
        print("STANDARD CLEANUP: Detected citeturn pattern")
        
        # First try basic regex replacement for standard cases
        old_citation_pattern = r'\s*citeturn\d+file\d+\.?'
        text = re.sub(old_citation_pattern, ".", text)
        
        # If still present, try more aggressive approaches
        if "citeturn" in text:
            # Handle special unicode characters that might be around the citation
            # Match any character (including Unicode) surrounding citeturn pattern
            unicode_citation_pattern = r'[^\w\s]*citeturn\d*file\d*[^\w\s]*'
            text = re.sub(unicode_citation_pattern, ".", text)
            
            # If still not removed, try brute force approach
            if "citeturn" in text:
                print("STANDARD CLEANUP: Using sentence-level split")
                parts = re.split(r'[^\w\s]*citeturn', text)
                if len(parts) > 1:
                    text = parts[0].rstrip()
                    if text and not text.endswith('.'):
                        text += '.'
    
    # Handle new-style citation format
    if "contentReference" in text:
        new_citation_pattern = r'\s*:?contentReference\[oaicite:\d+\]\{index=\d+\}\.?'
        text = re.sub(new_citation_pattern, ".", text)
    
    # Final emergency check - if still not clean, just cut at the citation marker
    if "citeturn" in text or "contentReference" in text:
        print("EMERGENCY CLEANUP: Still found citation markers, cutting text")
        citations = ["citeturn", "contentReference"]
        for citation in citations:
            if citation in text:
                idx = text.find(citation)
                if idx > 0:
                    text = text[:idx].rstrip()
                    if not text.endswith('.'):
                        text += '.'
    
    return text

def enhance_message_tone(text, has_extracted_farewell=False):
    """Enhance message with respectful tone for CEO audience."""
    if not text:
        return text
    
    # First check if text already has respectful opening
    respectful_openings = [
        "אני שמח להציג בפניך", "לרשותך", "בכבוד רב", "לשירותך",
        "אשמח לסייע", "בהתאם לבקשתך", "אני מתכבד", "אני שמח לעדכן"
    ]
    
    # Remove any existing duplicate greetings first (exact duplicates)
    duplicate_patterns = [
        r'(אני שמח לעדכן) אני שמח לעדכן'

    ]
    
    for pattern in duplicate_patterns:
        text = re.sub(pattern, r'\1', text)
    
    # Check if text already starts with a respectful opening
    has_respectful_opening = False
    for opening in respectful_openings:
        if opening in text[:150]:
            has_respectful_opening = True
            break
    
    # Only add respectful opening if it doesn't already have one
    if not has_respectful_opening:
        # Choose a respectful opening that fits the content
        if "?" in text[:50]:
            # If it's answering a question
            opening = "אני מתכבד להשיב לשאלתכם, "
        elif any(term in text.lower() for term in ["סטטוס", "מצב", "התקדמות"]):
            # If it's a status update
            opening = "אני שמח לעדכן כי "
        else:
            # Default respectful opening
            opening = "לרשותך, "
        
        # Add the opening to the first paragraph
        paragraphs = text.split("\n\n")
        if paragraphs and len(paragraphs[0]) > 0:
            # If the first character is a letter, make it lowercase after the greeting
            if paragraphs[0][0].isalpha():
                rest_of_text = paragraphs[0][1:] if len(paragraphs[0]) > 1 else ""
                paragraphs[0] = opening + paragraphs[0][0].lower() + rest_of_text
            else:
                paragraphs[0] = opening + paragraphs[0]
        elif paragraphs:
            paragraphs[0] = opening.strip()
            
        text = "\n\n".join(paragraphs)
    
    # Add respectful closing if there isn't one already AND we haven't extracted one
    if not has_extracted_farewell:
        respectful_closings = [
            "בכבוד רב", "אשמח לענות על שאלות נוספות", "לשירותך תמיד", 
            "אני לרשותכם", "בברכה", "בכל שאלה נוספת"
        ]
        
        has_respectful_closing = any(closing in text[-150:] for closing in respectful_closings)
        
        if not has_respectful_closing:
            text += "\n\nאשמח לענות על כל שאלה נוספת. בכבוד רב, עוזר הפרויקט."
    
    return text

def format_message_with_paragraphs(text):
    """Format message with proper paragraph structure while preserving original layout."""
    # Don't process empty text
    if not text:
        return text
    
    # First, normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Pre-process text to properly split numbered list items that appear on same line as previous content
    # Pattern: period, space, number, period, space, text - add newline after first period
    text = re.sub(r'(\.\s+)(\d+)(\.\s+)([^\n]+)', r'\1\n\2\3\4', text)
    
    # Also handle cases where there's a period, then a number, a period, and immediately text
    text = re.sub(r'(\.\s*)(\d+)(\.)(\S+)', r'\1\n\2\3 \4', text)
    
    # Split text into paragraphs based on double newlines
    paragraphs = text.split('\n\n')
    
    # Process each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # Process the paragraph
        paragraph = paragraph.strip()
        
        # Make sure paragraphs with numbered items (3. text) start on a new line
        paragraph = re.sub(r'(\.\s+)(\d+)(\.\s+)', r'\1\n\2\3', paragraph)
        
        processed_paragraphs.append(paragraph)
    
    # Join paragraphs with double newlines
    formatted_text = '\n\n'.join(processed_paragraphs)
    
    # Final clean-up - multiple periods, etc.
    formatted_text = re.sub(r'\.{2,}', '.', formatted_text)  # Replace multiple periods with a single one
    formatted_text = re.sub(r'\s+\.', '.', formatted_text)   # Remove spaces before periods
    
    return formatted_text


async def check_thread_health(thread_id: str) -> dict:
    """
    Check if a thread is healthy or potentially overloaded.
    Examines message count and thread age to determine if it should be rotated.
    """
    try:
        # Get message count
        message_count = await api_call(
            client.beta.threads.messages.list,
            "threads.messages.count",
            thread_id=thread_id,
            limit=100  # We only need the count, not all messages
        )
        
        # Check if we're approaching message limits
        # OpenAI doesn't document exact limits, but threads with many messages can become problematic
        count = len(message_count.data)
        logger.info(f"Thread {thread_id} has {count} messages")
        
        # Get thread details to check creation time
        thread = await api_call(
            client.beta.threads.retrieve,
            "threads.retrieve",
            thread_id=thread_id
        )
        
        # Check thread age (if older than 7 days, consider rotation)
        thread_created = thread.created_at
        current_time = int(time.time())
        thread_age_days = (current_time - thread_created) / (60 * 60 * 24)
        logger.info(f"Thread {thread_id} is {thread_age_days:.1f} days old")
        
        # Perform health check
        needs_rotation = False
        reasons = []
        
        # Check message count - if more than 50 messages, rotate thread
        if count > 50:
            needs_rotation = True
            reasons.append(f"Too many messages ({count} > 50)")
            
        # Check thread age - if older than 7 days, rotate thread  
        if thread_age_days > 7:
            needs_rotation = True
            reasons.append(f"Thread too old ({thread_age_days:.1f} days > 7)")
            
        return {
            "healthy": not needs_rotation,
            "message_count": count,
            "age_days": thread_age_days,
            "needs_rotation": needs_rotation,
            "reasons": reasons
        }
    except Exception as e:
        logger.error(f"Error checking thread health: {e}")
        # If we can't check health, assume we need a new thread to be safe
        return {
            "healthy": False,
            "needs_rotation": True,
            "reasons": [f"Error checking health: {str(e)}"]
        }

async def rotate_thread(project_id: str) -> str:
    """
    Create a new thread and update the project with the new thread ID.
    Returns the new thread ID.
    """
    try:
        logger.info(f"Rotating thread for project {project_id}")
        new_thread_id = await create_project_thread()
        if not new_thread_id:
            logger.error("Failed to create new thread for rotation")
            return None
            
        # Update project with new thread ID
        from database import update_project_thread
        update_project_thread(project_id, new_thread_id)
        logger.info(f"Thread rotated successfully. New thread ID: {new_thread_id}")
        return new_thread_id
    except Exception as e:
        logger.error(f"Error rotating thread: {e}")
        return None

async def get_ai_response(project: dict, user_message: str) -> str:
    """Process a user message and get an AI response using the Assistants API."""
    try:
        logger.info(f"Starting AI response generation for message: {user_message[:30]}...")
        
        assistant_id = project.get("assistant_id")
        thread_id = project.get("thread_id")
        
        if not assistant_id:
            logger.info(f"No assistant ID found for project {project.get('name', 'unknown')}. Creating new assistant.")
            assistant_id = await create_project_assistant(project["name"], project["type"], project["description"])
            if not assistant_id:
                logger.error("Failed to create assistant")
                return "Error creating assistant. Please try again later."
            from database import update_project_assistant
            update_project_assistant(str(project["_id"]), assistant_id)
            project["assistant_id"] = assistant_id
            logger.info(f"Created and updated assistant ID: {assistant_id}")
        
        if not thread_id:
            logger.info(f"No thread ID found for project {project.get('name', 'unknown')}. Creating new thread.")
            thread_id = await create_project_thread()
            if not thread_id:
                logger.error("Failed to create thread")
                return "Error creating thread. Please try again later."
            from database import update_project_thread
            update_project_thread(str(project["_id"]), thread_id)
            project["thread_id"] = thread_id
            logger.info(f"Created and updated thread ID: {thread_id}")
        else:
            # Check thread health and rotate if necessary
            health = await check_thread_health(thread_id)
            if health["needs_rotation"]:
                logger.warning(f"Thread {thread_id} needs rotation: {', '.join(health['reasons'])}")
                new_thread_id = await rotate_thread(str(project["_id"]))
                if new_thread_id:
                    thread_id = new_thread_id
                    project["thread_id"] = thread_id
                    logger.info(f"Using new thread ID: {thread_id}")
                else:
                    logger.error("Failed to rotate thread, continuing with existing thread")
        
        # Add message to thread
        message_id = await add_message_to_thread(thread_id, user_message, project["owner"])
        if not message_id:
            logger.error("Failed to add message to thread")
            return "Error sending message. Please try again later."
        
        # Run assistant with retry logic
        run_id = await retry_run_assistant(thread_id, assistant_id, max_retries=2)
        if not run_id:
            logger.error("Failed to run assistant after retries")
            return "שגיאה בהפעלת העוזר. אנא נסה שוב מאוחר יותר."
        
        # Implement exponential backoff for polling
        status = await get_run_status(thread_id, run_id)
        max_polls = 20  # Reduced from 30 to 20
        polls = 0
        base_delay = 0.5  # Start with 0.5 seconds
        max_delay = 2.0   # Maximum delay of 2 seconds
        run_error_msg = None
        
        logger.info(f"Polling for run completion. Initial status: {status}")
        
        while status not in ["completed", "failed", "cancelled", "expired"] and polls < max_polls:
            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** polls), max_delay)  # Exponential backoff with cap
            logger.info(f"Waiting {delay:.2f}s before next poll. Poll {polls+1}/{max_polls}")
            await asyncio.sleep(delay)
            
            status = await get_run_status(thread_id, run_id)
            polls += 1
            
            # If we get a completed status early, break immediately
            if status == "completed":
                logger.info("Run completed successfully")
                break
            
            # If run failed, get error details
            if status == "failed":
                run_details = await get_run_details(thread_id, run_id)
                if run_details.get('error') and run_details['error'].get('message'):
                    run_error_msg = run_details['error']['message']
                    logger.error(f"Run failed with error: {run_error_msg}")
                    
                    # If failure is potentially related to thread issues, rotate thread for next time
                    if "thread" in run_error_msg.lower() or "token" in run_error_msg.lower() or "limit" in run_error_msg.lower():
                        logger.warning(f"Failure may be thread-related, scheduling thread rotation")
                        await rotate_thread(str(project["_id"]))
                break
        
        # Handle different statuses
        if status == "failed":
            # Provide a more specific error message based on error details
            if run_error_msg:
                if "rate_limit" in run_error_msg.lower():
                    return "העוזר אינו זמין כרגע עקב עומס. אנא נסה שוב בעוד מספר דקות."
                elif "content_filter" in run_error_msg.lower():
                    return "לא ניתן לענות על השאלה בגלל מדיניות התוכן שלנו."
                elif "invalid" in run_error_msg.lower():
                    return "חלה שגיאה בעיבוד הבקשה. אנא נסה לנסח את השאלה באופן אחר."
                elif "token" in run_error_msg.lower() or "limit" in run_error_msg.lower():
                    # This is likely a thread size issue, so we've already rotated the thread for next time
                    return "חלה שגיאה בעיבוד הבקשה עקב מגבלות המערכת. אנא נסה שוב בעוד מספר רגעים."
                else:
                    return f"העוזר נתקל בשגיאה: {run_error_msg}"
            else:
                return "העוזר נתקל בשגיאה. אנא נסה שוב מאוחר יותר."
        
        if status == "cancelled":
            return "הבקשה בוטלה. אנא נסה שוב."
            
        if status == "expired":
            return "זמן העיבוד של הבקשה פג. אנא נסה שוב."
            
        if status != "completed":
            logger.warning(f"Run did not complete successfully. Final status: {status}")
            return f"העוזר מעבד את הנתונים במצב '{status}'. זמן העיבוד ארוך מהרגיל, אנא נסה שוב בעוד מספר רגעים."
        
        response = await get_run_result(thread_id)
        if not response:
            logger.warning("Run completed but no response was found")
            return "העוזר השלים את העיבוד אך לא סיפק תשובה. אנא נסה שוב."
            
        logger.info(f"AI response complete. Response length: {len(response)}")
        return response
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


async def get_run_details(thread_id: str, run_id: str) -> dict:
    """Get detailed information about a run, including error information if available."""
    try:
        run = await api_call(
            client.beta.threads.runs.retrieve,
            "threads.runs.retrieve_detailed",
            thread_id=thread_id,
            run_id=run_id
        )
        
        # Extract error information if available
        error_info = {}
        if hasattr(run, 'last_error') and run.last_error:
            error_info = {
                'code': getattr(run.last_error, 'code', 'unknown'),
                'message': getattr(run.last_error, 'message', 'Unknown error'),
            }
            logger.error(f"Run error details: {error_info}")
        
        return {
            'status': run.status,
            'error': error_info,
            'created_at': run.created_at,
            'completed_at': getattr(run, 'completed_at', None)
        }
    except Exception as e:
        logger.error(f"Error retrieving run details: {e}")
        return {
            'status': 'error',
            'error': {'code': 'client_error', 'message': str(e)}
        }

async def retry_run_assistant(thread_id: str, assistant_id: str, max_retries=2) -> str:
    """Run the assistant with retry logic for recoverable errors."""
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"Retry attempt {retries}/{max_retries} for running assistant")
                
            run = await api_call(
                client.beta.threads.runs.create,
                "threads.runs.create_with_retry",
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            logger.info(f"Started run (attempt {retries+1}): {run.id}")
            return run.id
        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Don't retry rate limit errors
            if "rate_limit" in error_msg:
                logger.error(f"Rate limit hit, not retrying: {e}")
                break
                
            # Don't retry certain error types that won't benefit from retrying
            if any(x in error_msg for x in ["authentication", "permission", "invalid_request_error"]):
                logger.error(f"Non-recoverable error, not retrying: {e}")
                break
                
            retries += 1
            if retries <= max_retries:
                # Exponential backoff
                wait_time = 2 ** retries
                logger.info(f"Recoverable error, retrying in {wait_time} seconds: {e}")
                await asyncio.sleep(wait_time)
    
    # If we get here, all retries failed or we hit a non-retryable error
    logger.error(f"All retries failed or non-retryable error encountered: {last_error}")
    return None

async def get_ai_response_stream(project: dict, user_message: str):
    """Process a user message and get an AI response using streaming."""
    try:
        logger.info(f"Starting streaming AI response for message: {user_message[:30]}...")
        
        assistant_id = project.get("assistant_id")
        thread_id = project.get("thread_id")
        
        if not assistant_id:
            logger.info(f"No assistant ID found for project {project.get('name', 'unknown')}. Creating new assistant.")
            assistant_id = await create_project_assistant(project["name"], project["type"], project["description"])
            if not assistant_id:
                logger.error("Failed to create assistant")
                yield "Error creating assistant. Please try again later."
                return
            from database import update_project_assistant
            update_project_assistant(str(project["_id"]), assistant_id)
            project["assistant_id"] = assistant_id
            logger.info(f"Created and updated assistant ID: {assistant_id}")
        
        if not thread_id:
            logger.info(f"No thread ID found for project {project.get('name', 'unknown')}. Creating new thread.")
            thread_id = await create_project_thread()
            if not thread_id:
                logger.error("Failed to create thread")
                yield "Error creating thread. Please try again later."
                return
            from database import update_project_thread
            update_project_thread(str(project["_id"]), thread_id)
            project["thread_id"] = thread_id
            logger.info(f"Created and updated thread ID: {thread_id}")
        else:
            # Check thread health and rotate if necessary
            health = await check_thread_health(thread_id)
            if health["needs_rotation"]:
                logger.warning(f"Thread {thread_id} needs rotation: {', '.join(health['reasons'])}")
                new_thread_id = await rotate_thread(str(project["_id"]))
                if new_thread_id:
                    thread_id = new_thread_id
                    project["thread_id"] = thread_id
                    logger.info(f"Using new thread ID: {thread_id}")
                else:
                    logger.error("Failed to rotate thread, continuing with existing thread")
        
        # Add message to thread
        message_id = await add_message_to_thread(thread_id, user_message, project["owner"])
        if not message_id:
            logger.error("Failed to add message to thread")
            yield "Error sending message. Please try again later."
            return
        
        # Run assistant with retry logic
        run_id = await retry_run_assistant(thread_id, assistant_id, max_retries=2)
        if not run_id:
            logger.error("Failed to run assistant after retries")
            yield "שגיאה בהפעלת העוזר. אנא נסה שוב מאוחר יותר."
            return
        
        # Poll for status and content
        status = await get_run_status(thread_id, run_id)
        max_polls = 30
        polls = 0
        base_delay = 0.5
        max_delay = 2.0
        last_message = ""
        initial_waiting = True
        accumulated_message = ""
        run_failed = False
        run_error_msg = None
        
        logger.info(f"Starting streaming polls. Initial status: {status}")
        
        while status not in ["completed", "failed", "cancelled", "expired"] and polls < max_polls:
            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** polls), max_delay)
            logger.info(f"Stream waiting {delay:.2f}s before next poll. Poll {polls+1}/{max_polls}")
            await asyncio.sleep(delay)
            
            status = await get_run_status(thread_id, run_id)
            polls += 1
            
            # If run failed, get detailed error information
            if status == "failed":
                run_failed = True
                run_details = await get_run_details(thread_id, run_id)
                if run_details.get('error') and run_details['error'].get('message'):
                    run_error_msg = run_details['error']['message']
                    logger.error(f"Run failed with error: {run_error_msg}")
                    
                    # If failure is potentially related to thread issues, rotate thread for next time
                    if "thread" in run_error_msg.lower() or "token" in run_error_msg.lower() or "limit" in run_error_msg.lower():
                        logger.warning(f"Failure may be thread-related, scheduling thread rotation")
                        await rotate_thread(str(project["_id"]))
                break
            
            # Get partial response if available
            current_message = await get_run_result(thread_id, ignore_empty=True)
            
            if current_message:
                logger.info(f"Received partial response of length {len(current_message)}")
                current_message = clean_citations(current_message)
                initial_waiting = False
            
            if current_message and current_message != last_message:
                new_content = current_message[len(last_message):]
                if new_content:
                    cleaned_content = clean_citations(new_content)
                    accumulated_message += cleaned_content
                    logger.info(f"Yielding new content of length {len(cleaned_content)}")
                    yield cleaned_content
                last_message = current_message
            
            if status == "completed":
                logger.info("Streaming run completed successfully")
                break
        
        # Handle different end states
        if status == "failed":
            run_failed = True
            # Get detailed error information if we haven't already
            if not run_error_msg:
                run_details = await get_run_details(thread_id, run_id)
                if run_details.get('error') and run_details['error'].get('message'):
                    run_error_msg = run_details['error']['message']
                    logger.error(f"Run failed with error: {run_error_msg}")
        
        if run_failed:
            # Provide a more specific error message based on error details
            if run_error_msg:
                if "rate_limit" in run_error_msg.lower():
                    yield "העוזר אינו זמין כרגע עקב עומס. אנא נסה שוב בעוד מספר דקות."
                elif "content_filter" in run_error_msg.lower():
                    yield "לא ניתן לענות על השאלה בגלל מדיניות התוכן שלנו."
                elif "invalid" in run_error_msg.lower():
                    yield "חלה שגיאה בעיבוד הבקשה. אנא נסה לנסח את השאלה באופן אחר."
                elif "token" in run_error_msg.lower() or "limit" in run_error_msg.lower():
                    # This is likely a thread size issue, so we've already rotated the thread for next time
                    yield "חלה שגיאה בעיבוד הבקשה עקב מגבלות המערכת. אנא נסה שוב בעוד מספר רגעים."
                else:
                    yield f"העוזר נתקל בשגיאה: {run_error_msg}"
            else:
                yield "העוזר נתקל בשגיאה. אנא נסה שוב מאוחר יותר."
            return
            
        if initial_waiting and status != "completed":
            if status == "expired":
                yield "זמן העיבוד של הבקשה פג. אנא נסה שוב."
            elif status == "cancelled":
                yield "העיבוד בוטל. אנא נסה שוב."
            else:
                yield "העוזר מעבד את הבקשה. זמן העיבוד ארוך מהרגיל, אנא נסה שוב בעוד מספר רגעים."
            return
            
        if status != "completed" and not initial_waiting:
            yield f"העוזר לא השלים את העיבוד, המצב הנוכחי: '{status}'. נא לנסות שוב."
            return
            
        # Final message
        final_message = await get_run_result(thread_id)
        if final_message:
            final_message = clean_citations(final_message)
            
            # Safety check
            if "citeturn" in final_message or "contentReference" in final_message:
                logger.warning("Final message still contains citation markers")
                for pattern in ["citeturn", "contentReference"]:
                    if pattern in final_message:
                        parts = final_message.split(pattern)
                        final_message = parts[0].rstrip() + "."
        
        if final_message and final_message != last_message:
            new_final_content = final_message[len(last_message):]
            new_final_content = clean_citations(new_final_content)
            logger.info(f"Yielding final content of length {len(new_final_content)}")
            yield new_final_content
        elif not final_message and status == "completed":
            # This should rarely happen - completed status but no message
            logger.warning("Run completed but no message was found")
            yield "העוזר השלים את העיבוד אך לא נמצאה תשובה. אנא נסה שוב."
            
    except Exception as e:
        logger.error(f"Error in streaming AI response: {str(e)}", exc_info=True)
        yield "אירעה שגיאה בעיבוד הבקשה. אנא נסה שוב מאוחר יותר."


async def create_message(thread_id: str, message: str) -> str:
    """Create a message in a thread."""
    try:
        loop = asyncio.get_running_loop()
        create_message_func = partial(
            client.beta.threads.messages.create,
            thread_id=thread_id,
            role="user",
            content=message
        )
        message_obj = await loop.run_in_executor(None, create_message_func)
        return message_obj.id
    except Exception as e:
        print(f"Error creating message: {e}")
        return None


async def wait_for_run_completion(thread_id: str, run_id: str, max_wait_time: int = 30) -> bool:
    """Wait for a run to complete with timeout."""
    try:
        max_polls = max_wait_time * 2  # Poll roughly every 0.5 seconds
        polls = 0
        while polls < max_polls:
            status = await get_run_status(thread_id, run_id)
            if status == "completed":
                return True
            if status in ["failed", "cancelled", "expired"]:
                print(f"Run ended with status: {status}")
                return False
            await asyncio.sleep(0.5)
            polls += 1
        return False
    except Exception as e:
        print(f"Error waiting for run completion: {e}")
        return False


async def get_response(thread_id: str) -> dict:
    """Get the latest response from a thread."""
    try:
        loop = asyncio.get_running_loop()
        get_messages_func = partial(
            client.beta.threads.messages.list,
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        messages = await loop.run_in_executor(None, get_messages_func)
        
        if messages.data and messages.data[0].role == "assistant":
            message_obj = messages.data[0]
            text_content = ""
            
            for content_item in message_obj.content:
                if content_item.type == "text":
                    text_content += content_item.text.value
            
            return {
                "text": text_content,
                "id": message_obj.id,
                "created_at": message_obj.created_at
            }
        return {"text": "", "id": None, "created_at": None}
    except Exception as e:
        print(f"Error getting response: {e}")
        return {"text": "", "error": str(e)}


async def get_project_overview(project: dict) -> dict:
    """
    Query the assistant to get an overview of the project.
    
    Returns a dictionary with the following keys:
    - contractor: מי הקבלן המבצע (Who is the contractor)
    - budget: מה תקציב הפרוייקט (What is the project budget)
    - completion_date: צפי לסיום פרוייקט (Expected project completion date)
    - status: סטטוס אחרון (Latest status)
    - plans: תוכניות להמשך (Future plans) - list
    - meetings: פגישות קרובות (Upcoming meetings) - list
    """
    try:
        assistant_id = project.get("assistant_id")
        thread_id = project.get("thread_id")
        
        # Create assistant if it doesn't exist (same logic as in get_ai_response)
        if not assistant_id:
            logger.info(f"No assistant found for project {project.get('name', 'unknown')}, creating new assistant")
            assistant_id = await create_project_assistant(project["name"], project["type"], project["description"])
            if not assistant_id:
                return {
                    "error": "שגיאה ביצירת עוזר. אנא נסה שוב מאוחר יותר."
                }
            from database import update_project_assistant
            update_project_assistant(str(project["_id"]), assistant_id)
            project["assistant_id"] = assistant_id
            logger.info(f"Created assistant {assistant_id} for project {project.get('name')}")
        
        # Create thread if it doesn't exist
        if not thread_id:
            logger.info(f"Creating new thread for project overview: {project.get('name', 'unknown')}")
            thread_id = await create_project_thread()
            if not thread_id:
                return {
                    "error": "שגיאה ביצירת שיחה. אנא נסה שוב מאוחר יותר."
                }
            # Update project with thread ID
            from database import update_project_thread
            update_project_thread(str(project["_id"]), thread_id)
            project["thread_id"] = thread_id
            logger.info(f"Thread created: {thread_id}")
        
        # Check thread health and rotate if necessary
        health = await check_thread_health(thread_id)
        if health["needs_rotation"]:
            logger.warning(f"Thread {thread_id} needs rotation: {', '.join(health['reasons'])}")
            new_thread_id = await rotate_thread(str(project["_id"]))
            if new_thread_id:
                thread_id = new_thread_id
                project["thread_id"] = thread_id
                logger.info(f"Using new thread ID: {thread_id} for project overview")
        
        logger.info(f"Getting project overview using assistant {assistant_id} and thread {thread_id}")
        
        # Create a message asking for project overview in Hebrew
        overview_prompt = """
        אנא ספק סקירה של הפרויקט הזה עם המידע הבא:
        1. מי הקבלן המבצע
        2. מה תקציב הפרוייקט
        3. צפי לסיום פרוייקט
        4. סטטוס אחרון
        5. תוכניות להמשך (רשימה של תוכניות)
        6. פגישות קרובות (רשימה של פגישות)
        
        אם אין לך מידע על אחד או יותר מהפרטים, אנא השב "לא ידוע" עבור אותו פרט.
        
        חשוב: אנא פורמט את התשובה שלך כאובייקט JSON תקין לחלוטין במבנה הבא, ללא תוספות לפני או אחרי ה-JSON:
        {
            "contractor": "שם הקבלן",
            "budget": "סכום התקציב",
            "completion_date": "תאריך סיום צפוי",
            "status": "סטטוס נוכחי",
            "plans": ["תוכנית 1", "תוכנית 2", "..."],
            "meetings": ["פגישה 1 בתאריך", "פגישה 2 בתאריך", "..."]
        }
        
        אנא התבסס רק על המידע הקיים בקבצים ובהיסטוריית השיחות. אל תמציא מידע.
        """
        
        # Send the message
        message_id = await create_message(thread_id, overview_prompt)
        if not message_id:
            logger.error("Failed to create message for project overview")
            return {
                "error": "שגיאה בשליחת בקשה לעוזר. אנא נסה שוב מאוחר יותר."
            }
        
        # Run the assistant with retry logic
        run_id = await retry_run_assistant(thread_id, assistant_id, max_retries=2)
        if not run_id:
            logger.error("Failed to run assistant for project overview")
            return {
                "error": "שגיאה בהפעלת העוזר. אנא נסה שוב מאוחר יותר."
            }
        
        logger.info(f"Started run {run_id} for project overview")
        
        # Poll for completion
        max_polls = 30
        poll_interval = 1.0
        
        for poll_count in range(max_polls):
            await asyncio.sleep(poll_interval)
            
            # Check run status
            run_status = await get_run_status(thread_id, run_id)
            logger.info(f"Run {run_id} status: {run_status} (poll {poll_count+1}/{max_polls})")
            
            if run_status == "completed":
                # Get the response
                response = await get_response(thread_id)
                
                if response and response.get("text"):
                    response_text = response.get("text")
                    logger.info(f"Received project overview response (length: {len(response_text)})")
                    
                    # Extract and parse JSON
                    try:
                        # Try to extract JSON from the response using multiple approaches
                        # 1. Try to find JSON in code blocks
                        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # 2. Try to find JSON object if not in code block
                            json_match = re.search(r'({[\s\S]*})', response_text)
                            if json_match:
                                json_str = json_match.group(1)
                            else:
                                json_str = response_text
                        
                        # Clean up the string to try to make it valid JSON
                        json_str = re.sub(r'^[^{]*', '', json_str)  # Remove anything before the first {
                        json_str = re.sub(r'[^}]*$', '', json_str)  # Remove anything after the last }
                        
                        # Handle specific JSON issues that might be present
                        # Fix trailing commas which are not valid in JSON
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        
                        # Replace single quotes with double quotes
                        json_str = json_str.replace("'", '"')
                        
                        # Fix unquoted keys - check for keys that aren't properly quoted
                        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                        
                        # Fix special characters like Hebrew in keys
                        json_str = re.sub(r'([{,]\s*)([^"\s:,{}[\]]+)\s*:', r'\1"\2":', json_str)
                        
                        logger.info(f"Attempting to parse JSON: {json_str[:100]}...")
                        
                        # Parse JSON
                        try:
                            overview_data = json.loads(json_str)
                        except json.JSONDecodeError as json_err:
                            logger.error(f"JSON parsing error: {json_err}")
                            
                            # Fallback approach: create default structure and extract values with regex
                            overview_data = {
                                "contractor": "לא ידוע",
                                "budget": "לא ידוע",
                                "completion_date": "לא ידוע",
                                "status": "לא ידוע",
                                "plans": [],
                                "meetings": []
                            }
                            
                            # Try to extract values for each field using regex
                            contractor_match = re.search(r'"contractor"\s*:\s*"([^"]*)"', response_text)
                            if contractor_match:
                                overview_data["contractor"] = contractor_match.group(1)
                                
                            budget_match = re.search(r'"budget"\s*:\s*"([^"]*)"', response_text)
                            if budget_match:
                                overview_data["budget"] = budget_match.group(1)
                                
                            date_match = re.search(r'"completion_date"\s*:\s*"([^"]*)"', response_text)
                            if date_match:
                                overview_data["completion_date"] = date_match.group(1)
                                
                            status_match = re.search(r'"status"\s*:\s*"([^"]*)"', response_text)
                            if status_match:
                                overview_data["status"] = status_match.group(1)
                                
                            # Try to extract array items
                            plans_list = re.findall(r'"plans"\s*:\s*\[\s*"([^"]*)"', response_text)
                            if plans_list:
                                overview_data["plans"] = plans_list
                                
                            # Better approach to extract all array items
                            plans_match = re.search(r'"plans"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
                            if plans_match:
                                plans_content = plans_match.group(1).strip()
                                # Extract each item from the array with a more robust pattern
                                # This pattern handles various formats including items with commas and periods
                                all_plans = re.findall(r'"((?:[^"\\]|\\.)*)(?:"|$)', plans_content)
                                if all_plans:
                                    # Filter out empty strings and clean up any escape characters
                                    all_plans = [plan.replace('\\"', '"').strip() for plan in all_plans if plan.strip()]
                                    if all_plans:
                                        overview_data["plans"] = all_plans
                                
                            meetings_list = re.findall(r'"meetings"\s*:\s*\[\s*"([^"]*)"', response_text)
                            if meetings_list:
                                overview_data["meetings"] = meetings_list
                                
                            # Better approach to extract all meeting items
                            meetings_match = re.search(r'"meetings"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
                            if meetings_match:
                                meetings_content = meetings_match.group(1).strip()
                                # Use the same improved pattern for meetings
                                all_meetings = re.findall(r'"((?:[^"\\]|\\.)*)(?:"|$)', meetings_content)
                                if all_meetings:
                                    # Filter out empty strings and clean up any escape characters
                                    all_meetings = [meeting.replace('\\"', '"').strip() for meeting in all_meetings if meeting.strip()]
                                    if all_meetings:
                                        overview_data["meetings"] = all_meetings
                        
                        # Ensure all required keys exist
                        required_keys = ["contractor", "budget", "completion_date", "status", "plans", "meetings"]
                        for key in required_keys:
                            if key not in overview_data:
                                overview_data[key] = "לא ידוע"
                        
                        # Ensure plans and meetings are lists
                        if not isinstance(overview_data["plans"], list):
                            overview_data["plans"] = [overview_data["plans"]] if overview_data["plans"] else []
                        
                        if not isinstance(overview_data["meetings"], list):
                            overview_data["meetings"] = [overview_data["meetings"]] if overview_data["meetings"] else []
                        
                        logger.info(f"Successfully parsed overview data with {len(overview_data['plans'])} plans and {len(overview_data['meetings'])} meetings")
                        return overview_data
                    except Exception as e:
                        logger.error(f"Error parsing JSON from assistant response: {e}", exc_info=True)
                        return {
                            "error": f"שגיאה בניתוח תשובת העוזר: {str(e)}"
                        }
                else:
                    logger.warning("Received empty response from assistant")
                    return {
                        "error": "לא התקבלה תשובה מהעוזר המלאכותי"
                    }
            elif run_status == "failed":
                # Get detailed error information
                run_details = await get_run_details(thread_id, run_id)
                error_msg = "סיבה לא ידועה"
                if run_details.get('error') and run_details['error'].get('message'):
                    error_msg = run_details['error']['message']
                
                logger.error(f"Run failed with error: {error_msg}")
                return {
                    "error": f"עיבוד נכשל: {error_msg}"
                }
            elif run_status == "expired":
                logger.warning(f"Run {run_id} expired")
                return {
                    "error": "עיבוד נכשל עקב פסק זמן"
                }
                
        # If we've reached here, the run is still processing
        logger.warning(f"Reached maximum polls ({max_polls}) for run {run_id}, still in status: {run_status}")
        return {
            "error": "העוזר המלאכותי עדיין מעבד את הבקשה. אנא נסה שוב בעוד מספר רגעים."
        }
        
    except Exception as e:
        logger.error(f"Error getting project overview: {e}", exc_info=True)
        return {
            "error": f"שגיאה: {str(e)}"
        }


async def get_project_graph_data(project: dict) -> dict:
    """
    Generate visual graph data for the project based on the project information and AI assistant insights.
    
    Returns a dictionary with the following keys:
    - budget_distribution: Data for pie chart showing budget distribution by category
    - progress_timeline: Data for line chart showing project progress over time
    - task_status: Data for bar chart showing tasks by status
    - completion_forecast: Data for line chart showing completion forecast
    """
    try:
        assistant_id = project.get("assistant_id")
        thread_id = project.get("thread_id")
        
        # Create assistant if it doesn't exist (same logic as in get_ai_response)
        if not assistant_id:
            logger.info(f"No assistant found for project {project.get('name')}, creating new assistant")
            assistant_id = await create_project_assistant(project["name"], project["type"], project["description"])
            if not assistant_id:
                return {
                    "error": "שגיאה ביצירת עוזר. אנא נסה שוב מאוחר יותר."
                }
            from database import update_project_assistant
            update_project_assistant(str(project["_id"]), assistant_id)
            project["assistant_id"] = assistant_id
            logger.info(f"Created assistant {assistant_id} for project {project.get('name')}")
        
        # Create thread if it doesn't exist
        if not thread_id:
            logger.info(f"No thread found for project {project.get('name')}, creating new thread")
            thread_id = await create_project_thread()
            if not thread_id:
                return {
                    "error": "שגיאה ביצירת שיחה. אנא נסה שוב מאוחר יותר."
                }
            from database import update_project_thread
            update_project_thread(str(project["_id"]), thread_id)
            project["thread_id"] = thread_id
            logger.info(f"Created thread {thread_id} for project {project.get('name')}")
        
        # Check thread health and rotate if needed
        health = await check_thread_health(thread_id)
        if health["needs_rotation"]:
            logger.warning(f"Thread {thread_id} needs rotation: {', '.join(health['reasons'])}")
            new_thread_id = await rotate_thread(str(project["_id"]))
            if new_thread_id:
                thread_id = new_thread_id
                project["thread_id"] = thread_id
                logger.info(f"Using new thread ID: {thread_id} for graph data")
        
        # Ask the assistant to generate graph data for the project
        instructions = """
        אני צריך מידע לצורך תצוגה חזותית על הפרויקט. השב בפורמט JSON בלבד (ללא הסברים או טקסט נוסף) שאפשר לפרסר באופן אוטומטי.
        הפורמט הנדרש הוא:
        {
            "budget_distribution": [
                {"category": "שם קטגוריה", "amount": מספר}
            ],
            "progress_timeline": [
                {"date": "תאריך", "percentage": מספר}
            ],
            "task_status": [
                {"status": "שם סטטוס", "count": מספר}
            ],
            "completion_forecast": [
                {"date": "תאריך", "forecast": מספר, "actual": מספר}
            ]
        }
        
        budget_distribution: חלוקת התקציב לפי קטגוריות שונות (הוצ' חומרים, כח אדם, אישורים, וכו')
        progress_timeline: התקדמות הפרויקט לאורך זמן (באחוזים)
        task_status: מספר המשימות לפי סטטוס (הושלמו, בביצוע, מתוכננות, וכו')
        completion_forecast: השוואה בין התקדמות בפועל לתחזית ההתקדמות
        
        חשוב: התבסס אך ורק על המידע שיש לך על הפרויקט. אם אין מספיק מידע, ספק נתונים הגיוניים שמייצגים פרויקט מסוג זה בשלב שבו הוא נמצא.
        """
        
        # Create and submit the message
        message_id = await create_message(thread_id, instructions)
        if not message_id:
            return {
                "error": "שגיאה בשליחת הוראות לעוזר. אנא נסה שוב מאוחר יותר."
            }
        
        # Run the assistant
        run_id = await retry_run_assistant(thread_id, assistant_id, max_retries=2)
        if not run_id:
            return {
                "error": "שגיאה בהפעלת העוזר. אנא נסה שוב מאוחר יותר."
            }
        
        # Wait for completion
        completed = await wait_for_run_completion(thread_id, run_id, max_wait_time=30)
        if not completed:
            # Check run status for more informative error
            run_details = await get_run_details(thread_id, run_id)
            if run_details.get('status') == "failed" and run_details.get('error'):
                error_msg = run_details['error'].get('message', "העוזר לא השלים את הניתוח.")
                return {"error": f"{error_msg} אנא נסה שוב מאוחר יותר."}
            
            return {
                "error": "העוזר לא השלים את הניתוח בזמן סביר. אנא נסה שוב מאוחר יותר."
            }
        
        # Get the response
        response = await get_response(thread_id)
        
        if not response or not response.get("text", ""):
            return {
                "error": "לא התקבלו נתונים מהעוזר המלאכותי. אנא נסה שוב מאוחר יותר."
            }
            
        # Parse the JSON response
        try:
            response_text = response.get("text", "")
            
            # Extract JSON part if there's text around it
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try to find a JSON object directly
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    response_text = json_match.group(1)
            
            # Parse the JSON
            graph_data = json.loads(response_text)
            
            # Ensure all required keys exist
            required_keys = ["budget_distribution", "progress_timeline", "task_status", "completion_forecast"]
            for key in required_keys:
                if key not in graph_data or not graph_data[key]:
                    # Initialize with empty array if key is missing or value is None
                    graph_data[key] = []
                elif not isinstance(graph_data[key], list):
                    # Convert to list if somehow not a list (additional safety)
                    graph_data[key] = [graph_data[key]] if graph_data[key] else []
            
            return graph_data
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing graph data: {e}", exc_info=True)
            logger.error(f"Response text: {response.get('text', '')}")
            
            return {
                "error": "שגיאה בעיבוד נתוני הגרפים. אנא נסה שוב מאוחר יותר."
            }
        
    except Exception as e:
        logger.error(f"Error generating graph data: {e}", exc_info=True)
        return {"error": f"שגיאה בעיבוד הנתונים: {str(e)}"}


async def process_user_request(project, user_message):
    async for chunk in get_ai_response_stream(project, user_message):
        print(chunk, end='')
