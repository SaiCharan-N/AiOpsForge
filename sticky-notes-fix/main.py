from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
from pathlib import Path

app = FastAPI()

# Setup directories
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory storage for notes
notes_storage = {}
note_counter = 0


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page showing all notes"""
    return templates.TemplateResponse("index.html", {"request": request, "notes": notes_storage})


@app.post("/notes")
async def create_note(title: str = Form(...), content: str = Form(...)):
    """Create a new sticky note"""
    global note_counter
    note_counter += 1
    note_id = note_counter
    notes_storage[note_id] = {
        "id": note_id,
        "title": title,
        "content": content,
        "color": "yellow"
    }
    return {"id": note_id, "message": "Note created successfully"}


@app.get("/notes/{note_id}", response_class=HTMLResponse)
async def view_note(note_id: int, request: Request):
    """View a single note"""
    if note_id not in notes_storage:
        raise HTTPException(status_code=404, detail="Note not found")
    note = notes_storage[note_id]
    return templates.TemplateResponse("note.html", {"request": request, "note": note})


@app.post("/notes/{note_id}")
async def update_note(note_id: int, title: str = Form(...), content: str = Form(...)):
    """Update an existing note"""
    if note_id not in notes_storage:
        raise HTTPException(status_code=404, detail="Note not found")
    notes_storage[note_id]["title"] = title
    notes_storage[note_id]["content"] = content
    return {"message": "Note updated successfully"}


@app.delete("/notes/{note_id}")
async def delete_note(note_id: int):
    """Delete a note"""
    if note_id not in notes_storage:
        raise HTTPException(status_code=404, detail="Note not found")
    del notes_storage[note_id]
    return {"message": "Note deleted successfully"}


@app.get("/api/notes")
async def get_all_notes():
    """Get all notes as JSON"""
    return {"notes": list(notes_storage.values())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
