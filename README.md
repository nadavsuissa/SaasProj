# Project Management System

A full-featured project management system with AI assistant integration for construction projects.

## Features

- User authentication and account management
- Project creation and management
- File upload and management with OpenAI integration
- Real-time chat with AI assistant powered by OpenAI
- Project analytics and reporting

## Requirements

- Python 3.8+
- MongoDB
- OpenAI API Key

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the `.env.template`:
   ```
   cp .env.template .env
   ```
   Then edit the `.env` file with your actual credentials.

## Configuration

Edit the `.env` file to configure the application:

- `SECRET_KEY`: A secret key for session security
- `MONGO_CONNECTION_STRING`: MongoDB connection string
- `OPENAI_API_KEY`: Your OpenAI API key
- Additional server settings if needed

## Running the Application

Start the application with:

```
uvicorn main:app --reload
```

The application will be available at http://localhost:8000

## API Documentation

API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Directory Structure

```
.
├── main.py           # FastAPI application
├── ai_helper.py      # OpenAI integration
├── database.py       # MongoDB integration
├── templates/        # HTML templates
├── static/           # Static files (CSS, JS)
├── uploads/          # Uploaded files
├── requirements.txt  # Python dependencies
└── .env              # Environment variables
```

## Security Notes

- Never commit your `.env` file or any files containing credentials
- Keep your OpenAI API key and MongoDB credentials secure
- Use proper authentication for all routes

## License

[License Information] 