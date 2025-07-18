# 🏆 Mix Tournament Web App - Architecture Plan

## 🎯 **CONCEPT:**

**AI Model Battle Arena** where musicians upload tracks, AI models compete in head-to-head mixing battles, users vote, and losing models evolve by learning from winners!

---

## 🏗️ **FOLDER STRUCTURE:**

### **1. Backend (`tournament_webapp/backend/`)**

```
backend/
├── app.py                    # FastAPI main server
├── tournament_engine.py      # Core tournament logic
├── model_evolution.py        # AI model weight mixing & training
├── audio_processor.py        # Audio mixing using existing AI
├── database_models.py        # User, Tournament, Vote schemas
├── api/
│   ├── tournaments.py        # Tournament CRUD endpoints
│   ├── models.py            # AI model management
│   ├── audio.py             # Audio upload/processing
│   └── users.py             # User management & sharing
└── requirements.txt         # Dependencies
```

### **2. Frontend (`tournament_webapp/frontend/`)**

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── AudioPlayer.jsx     # A/B comparison player
│   │   ├── TournamentArena.jsx # Battle interface
│   │   ├── ModelCard.jsx       # AI model display
│   │   ├── UploadZone.jsx      # Drag-drop audio upload
│   │   └── ShareModal.jsx      # Social sharing component
│   ├── pages/
│   │   ├── Tournament.jsx      # Main tournament page
│   │   ├── Results.jsx         # Tournament results
│   │   └── Shared.jsx          # Shared tournament view
│   ├── hooks/
│   │   ├── useTournament.js    # Tournament state management
│   │   └── useAudio.js         # Audio playback logic
│   └── App.jsx
├── package.json
└── README.md
```

### **3. Tournament Models (`tournament_webapp/tournament_models/`)**

```
tournament_models/
├── base_competitors/         # Links to ../models/ champions
├── evolved/                 # User-generated hybrid models
│   ├── user_123_hybrid_v1.pth
│   ├── user_456_hybrid_v2.pth
│   └── community_favorites/
├── user_preferences/        # Personalized models
└── model_genealogy.json    # Track model evolution history
```

### **4. Database (`tournament_webapp/database/`)**

```
database/
├── tournaments.db          # SQLite database
├── init_db.py             # Database setup
└── schemas/
    ├── tournaments.sql    # Tournament records
    ├── votes.sql          # User voting data
    ├── models.sql         # AI model metadata
    └── users.sql          # User accounts & sharing
```

### **5. Static Assets (`tournament_webapp/static/`)**

```
static/
├── uploads/               # User-uploaded audio
├── processed/            # Tournament-mixed audio
├── thumbnails/           # Audio waveform images
└── exports/              # Downloadable results
```

---

## 🎛️ **MODEL STRATEGY:**

### **Starting Champions (Use Existing Models):**

1. **Baseline CNN** (`../models/baseline_cnn.pth`) → **"Classic Fighter"**
2. **Enhanced CNN** (`../models/enhanced_cnn.pth`) → **"Modern Warrior"**
3. **Improved Baseline** (`../models/improved_baseline_cnn.pth`) → **"Veteran Champion"**
4. **Weighted Ensemble** (`../models/weighted_ensemble.pth`) → **"Boss Fighter"** (your best)

### **Model Evolution Process:**

1. **Battle**: Two models mix the same track
2. **Vote**: User chooses winner
3. **Evolution**: Losing model = `0.7 * winner_weights + 0.3 * loser_weights`
4. **Generation**: New hybrid model created for next battle
5. **Learning**: System learns user preferences across all votes

### **User Preference Learning:**

- Track voting patterns per user
- Generate personalized models
- Community models from popular vote patterns
- Genre-specific evolution paths

---

## 🚀 **TOURNAMENT FLOW:**

### **1. Upload Phase:**

```
User uploads → Audio analysis → Genre detection → Model selection
```

### **2. Tournament Phase (5 rounds):**

```
Round 1: Model A vs Model B → User votes → Winner advances
Round 2: Winner vs Model C → User votes → Winner advances
Round 3: Winner vs Model D → User votes → Winner advances
Round 4: Winner vs Hybrid(loser_weights) → User votes → Winner advances
Round 5: Winner vs Boss Model → Final result
```

### **3. Social Phase:**

```
Share result → Friends get 5 free tournaments → Community learning
```

---

## 🎵 **INTEGRATION WITH EXISTING CODE:**

### **Reuse Current Assets:**

- **`src/production_ai_mixer.py`** → Audio processing engine
- **`src/enhanced_musical_intelligence.py`** → Musical analysis
- **`models/*.pth`** → Starting tournament competitors
- **`src/ai_mixer.py`** → Core mixing logic

### **New Tournament-Specific Code:**

- **Model weight interpolation** for creating hybrids
- **User preference tracking** and learning
- **Tournament state management**
- **Web interface** for A/B comparison
- **Social sharing** and community features

---

## 🎯 **NEXT STEPS:**

1. **✅ Folder structure created**
2. **🔄 Build tournament engine** (model evolution logic)
3. **🔄 Create FastAPI backend** (tournament API)
4. **🔄 Build React frontend** (battle interface)
5. **🔄 Integrate existing AI models**
6. **🔄 Add user preference learning**
7. **🔄 Implement social sharing**

**Ready to start building the tournament engine?** 🏆
