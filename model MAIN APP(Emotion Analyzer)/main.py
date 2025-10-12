# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn
from transformers import pipeline
import random

app = FastAPI(title="Mental Health App API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load emotion analysis model
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# In-memory data store to replace Firestore
mood_entries_db = {}

# Data models
class MoodEntry(BaseModel):
    mood: str
    journal: Optional[str] = ""
    timestamp: datetime = datetime.now()
    user_id: str

class CopingSuggestion(BaseModel):
    suggestion: str
    category: str
    is_social: bool = False
    friend_name: Optional[str] = None
    activity: Optional[str] = None

# Emotion analysis endpoint
@app.post("/predict", response_model=dict)
async def predict_mood_and_suggestions(entry: MoodEntry):
    """
    Analyze journal text and return mood prediction with personalized coping suggestions
    """
    try:
        # Analyze journal text if provided
        emotion_scores = {}
        if entry.journal:
            emotions = emotion_model(entry.journal)
            emotion_scores = {emotion['label']: emotion['score'] for emotion in emotions[0]}
        
        # Get personalized coping suggestions
        suggestions = get_coping_suggestions(entry.mood)
        
        # Store in our local database
        if entry.user_id not in mood_entries_db:
            mood_entries_db[entry.user_id] = []
        mood_entries_db[entry.user_id].append(entry.dict())
        
        return {
            "mood": entry.mood,
            "emotion_analysis": emotion_scores,
            "coping_suggestions": suggestions,
            "entry_id": f"local-entry-{len(mood_entries_db[entry.user_id])}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_coping_suggestions(mood: str) -> List[CopingSuggestion]:
    """Generate personalized coping suggestions based on mood"""
    # Simplified without friends list
    
    suggestions_map = {
        "excellent": [
            CopingSuggestion(suggestion="Share your positive energy with someone today", category="social"),
            CopingSuggestion(suggestion="Practice gratitude - write down 3 things you're thankful for", category="mindfulness")
        ],
        "good": [
            CopingSuggestion(suggestion="Take a mindful walk outside", category="physical")
        ],
        "okay": [
            CopingSuggestion(suggestion="Practice deep breathing exercises", category="mindfulness")
        ],
        "struggling": [
            CopingSuggestion(suggestion="Try progressive muscle relaxation", category="mindfulness")
        ],
        "difficult": [
            CopingSuggestion(suggestion="Consider professional support", category="professional")
        ]
    }
    
    return suggestions_map.get(mood, [])

# Get mood history
@app.get("/mood-history/{user_id}")
async def get_mood_history(user_id: str, limit: int = 10):
    """Get user's mood history from local database"""
    history = mood_entries_db.get(user_id, [])
    # Sort by timestamp (most recent first) and apply limit
    sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    return {"history": sorted_history}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
