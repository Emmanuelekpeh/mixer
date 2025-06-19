"""
Database models and configuration for the Tournament Webapp
Provides persistent storage for tournaments, users, models, and battle results
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import json
import os
from pathlib import Path

# Database configuration
DATABASE_DIR = Path(__file__).parent / "data"
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/tournament.db"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """User model for tournament participants"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # User stats and profile
    tournaments_completed = Column(Integer, default=0)
    tournaments_won = Column(Integer, default=0)
    total_battles = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    tier = Column(String(20), default="Rookie")
    experience_points = Column(Integer, default=0)
    
    # Profile data stored as JSON
    profile_data = Column(JSON, default={})
    preferences = Column(JSON, default={})
    
    # Referral system
    referral_code = Column(String(20), unique=True)
    referred_by = Column(String, ForeignKey("users.id"), nullable=True)
    referral_count = Column(Integer, default=0)
    
    # Relationships
    tournaments = relationship("Tournament", back_populates="user")
    votes = relationship("BattleVote", back_populates="user")
    referrals = relationship("User")

class Tournament(Base):
    """Tournament model for AI mixing competitions"""
    __tablename__ = "tournaments"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Tournament configuration
    max_rounds = Column(Integer, default=5)
    current_round = Column(Integer, default=1)
    current_pair = Column(Integer, default=0)
    status = Column(String(20), default="active")  # active, completed, cancelled
    
    # Audio file information
    original_audio_file = Column(String(255), nullable=True)
    original_audio_url = Column(String(500), nullable=True)
    final_mix_url = Column(String(500), nullable=True)
    final_mix_download_url = Column(String(500), nullable=True)
    
    # Tournament results
    victor_model_id = Column(String, ForeignKey("ai_models.id"), nullable=True)
    
    # Tournament data stored as JSON for flexibility
    tournament_data = Column(JSON, default={})
    pairs_data = Column(JSON, default=[])
    battle_history = Column(JSON, default=[])
    
    # Performance metrics
    total_battles = Column(Integer, default=0)
    completion_rate = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="tournaments")
    victor_model = relationship("AIModel", foreign_keys=[victor_model_id])
    votes = relationship("BattleVote", back_populates="tournament")

class AIModel(Base):
    """AI Model information and statistics"""
    __tablename__ = "ai_models"
    
    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    nickname = Column(String(100), nullable=True)
    architecture = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Model metadata
    generation = Column(Integer, default=1)
    tier = Column(String(20), default="Amateur")
    model_file_path = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    
    # Performance metrics
    elo_rating = Column(Float, default=1200.0)
    total_battles = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # Model capabilities and preferences
    specializations = Column(JSON, default=[])
    preferred_genres = Column(JSON, default=[])
    signature_techniques = Column(JSON, default=[])
    capabilities = Column(JSON, default={})
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships - specify foreign_keys to avoid ambiguity
    tournament_victories = relationship("Tournament", foreign_keys="Tournament.victor_model_id", overlaps="victor_model")
    votes_received = relationship("BattleVote", foreign_keys="BattleVote.winning_model_id", overlaps="winning_model")

class BattleVote(Base):
    """Individual battle votes and results"""
    __tablename__ = "battle_votes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    tournament_id = Column(String, ForeignKey("tournaments.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Battle participants
    model_a_id = Column(String, ForeignKey("ai_models.id"), nullable=False)
    model_b_id = Column(String, ForeignKey("ai_models.id"), nullable=False)
    winning_model_id = Column(String, ForeignKey("ai_models.id"), nullable=False)
    
    # Vote details
    confidence = Column(Float, default=0.8)  # 0.0 to 1.0
    round_number = Column(Integer, nullable=False)
    pair_number = Column(Integer, nullable=False)
    
    # Audio files for this specific battle
    audio_a_url = Column(String(500), nullable=True)
    audio_b_url = Column(String(500), nullable=True)
    
    # Vote metadata
    vote_data = Column(JSON, default={})
    listening_time = Column(Float, nullable=True)  # Time spent listening before voting
    
    # Relationships
    tournament = relationship("Tournament", back_populates="votes")
    user = relationship("User", back_populates="votes")
    winning_model = relationship("AIModel", foreign_keys=[winning_model_id], overlaps="votes_received")

class TournamentAnalytics(Base):
    """Analytics and metrics for tournament performance"""
    __tablename__ = "tournament_analytics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Daily/periodic metrics
    tournaments_created = Column(Integer, default=0)
    tournaments_completed = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    
    # Popular models and architectures
    top_models = Column(JSON, default=[])
    top_architectures = Column(JSON, default=[])
    
    # Performance metrics
    average_completion_rate = Column(Float, default=0.0)
    average_battle_time = Column(Float, default=0.0)
    
    # System metrics
    analytics_data = Column(JSON, default={})

# Database utility functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables and default data"""
    print("🗄️  Initializing tournament database...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session for initial data
    db = SessionLocal()
    
    try:
        # Check if we need to seed data
        user_count = db.query(User).count()
        model_count = db.query(AIModel).count()
        
        if user_count == 0:
            print("  📝 Creating default demo user...")
            demo_user = User(
                id="demo_user",
                username="Demo User",
                email="demo@aimixer.com",
                tier="Professional",
                tournaments_completed=5,
                total_battles=25,
                referral_code="DEMO2025"            )
            db.add(demo_user)
        
        if model_count == 0:
            print("  🤖 Creating comprehensive AI model collection...")
            default_models = [
                # CNN-Based Models
                {
                    "id": "baseline_cnn",
                    "name": "Baseline CNN",
                    "nickname": "The Foundation",
                    "architecture": "CNN",
                    "tier": "Amateur",
                    "elo_rating": 1200,
                    "specializations": ["spectral_analysis"],
                    "capabilities": {
                        "spectral_analysis": 0.7,
                        "dynamic_range": 0.6,
                        "stereo_imaging": 0.5
                    }
                },
                {
                    "id": "enhanced_cnn", 
                    "name": "Enhanced CNN",
                    "nickname": "The Enhancer",
                    "architecture": "Enhanced CNN",
                    "tier": "Professional",
                    "elo_rating": 1350,
                    "specializations": ["dynamic_processing", "stereo_imaging"],
                    "capabilities": {
                        "spectral_analysis": 0.8,
                        "dynamic_range": 0.8,
                        "stereo_imaging": 0.9
                    }
                },
                {
                    "id": "deep_cnn",
                    "name": "Deep CNN Pro",
                    "nickname": "The Deep Diver",
                    "architecture": "Deep CNN",
                    "tier": "Professional",
                    "elo_rating": 1320,
                    "specializations": ["frequency_separation", "noise_reduction"],
                    "capabilities": {
                        "spectral_analysis": 0.85,
                        "dynamic_range": 0.75,
                        "noise_reduction": 0.9
                    }
                },
                # Transformer Models
                {
                    "id": "ast_transformer",
                    "name": "AST Transformer",
                    "nickname": "The Attention Master",
                    "architecture": "Transformer",
                    "tier": "Expert",
                    "elo_rating": 1500,
                    "specializations": ["harmonic_enhancement", "vocal_clarity"],
                    "capabilities": {
                        "spectral_analysis": 0.9,
                        "dynamic_range": 0.7,
                        "vocal_clarity": 0.95,
                        "harmonic_enhancement": 0.9
                    }
                },
                {
                    "id": "wav2vec_transformer",
                    "name": "Wav2Vec Transformer",
                    "nickname": "The Vector Virtuoso",
                    "architecture": "Transformer",
                    "tier": "Expert",
                    "elo_rating": 1480,
                    "specializations": ["temporal_modeling", "rhythm_enhancement"],
                    "capabilities": {
                        "temporal_modeling": 0.95,
                        "rhythm_enhancement": 0.9,
                        "dynamic_range": 0.8
                    }
                },
                # Hybrid Models
                {
                    "id": "cnn_transformer_hybrid",
                    "name": "CNN-Transformer Hybrid",
                    "nickname": "The Fusion Fighter",
                    "architecture": "Hybrid",
                    "tier": "Expert",
                    "elo_rating": 1520,
                    "specializations": ["multi_scale_analysis", "adaptive_mixing"],
                    "capabilities": {
                        "spectral_analysis": 0.9,
                        "temporal_modeling": 0.85,
                        "adaptive_mixing": 0.9,
                        "stereo_imaging": 0.85
                    }
                },
                # Advanced Models
                {
                    "id": "diffusion_mixer",
                    "name": "Diffusion Mixer",
                    "nickname": "The Probabilistic Perfectionist",
                    "architecture": "Diffusion",
                    "tier": "Legend",
                    "elo_rating": 1600,
                    "specializations": ["generative_mixing", "style_transfer"],
                    "capabilities": {
                        "generative_mixing": 0.95,
                        "style_transfer": 0.9,
                        "dynamic_range": 0.85,
                        "harmonic_enhancement": 0.85
                    }
                },
                {
                    "id": "neural_ode_mixer",
                    "name": "Neural ODE Mixer",
                    "nickname": "The Continuous Champion",
                    "architecture": "Neural ODE",
                    "tier": "Legend",
                    "elo_rating": 1580,
                    "specializations": ["continuous_dynamics", "flow_modeling"],
                    "capabilities": {
                        "continuous_dynamics": 0.95,
                        "flow_modeling": 0.9,
                        "temporal_modeling": 0.9,
                        "adaptive_mixing": 0.8
                    }
                },
                # Specialized Models
                {
                    "id": "vocal_specialist",
                    "name": "Vocal Specialist AI",
                    "nickname": "The Voice Whisperer",
                    "architecture": "Specialized CNN",
                    "tier": "Professional",
                    "elo_rating": 1400,
                    "specializations": ["vocal_clarity", "vocal_separation"],
                    "capabilities": {
                        "vocal_clarity": 0.95,
                        "vocal_separation": 0.9,
                        "harmonic_enhancement": 0.8
                    }
                },
                {
                    "id": "bass_master",
                    "name": "Bass Master AI",
                    "nickname": "The Low-End Legend",
                    "architecture": "Specialized CNN",
                    "tier": "Professional",
                    "elo_rating": 1380,
                    "specializations": ["bass_enhancement", "sub_harmonics"],
                    "capabilities": {
                        "bass_enhancement": 0.95,
                        "sub_harmonics": 0.9,
                        "dynamic_range": 0.85
                    }
                },
                {
                    "id": "mastering_ai",
                    "name": "Mastering AI Pro",
                    "nickname": "The Final Touch",
                    "architecture": "Multi-Stage CNN",
                    "tier": "Expert",
                    "elo_rating": 1460,
                    "specializations": ["mastering", "loudness_optimization"],
                    "capabilities": {
                        "mastering": 0.9,
                        "loudness_optimization": 0.95,
                        "dynamic_range": 0.9,
                        "stereo_imaging": 0.85
                    }
                },
                # Experimental Models
                {
                    "id": "quantum_mixer",
                    "name": "Quantum Mixer",
                    "nickname": "The Quantum Leap",
                    "architecture": "Quantum-Inspired",
                    "tier": "Legend",
                    "elo_rating": 1650,
                    "specializations": ["quantum_superposition", "entangled_mixing"],
                    "capabilities": {
                        "quantum_superposition": 0.9,
                        "entangled_mixing": 0.85,
                        "spectral_analysis": 0.9,
                        "adaptive_mixing": 0.9
                    }
                }
            ]
            
            for model_data in default_models:
                model = AIModel(**model_data)
                db.add(model)
        
        db.commit()
        print("  ✅ Database initialized successfully!")
        
    except Exception as e:
        print(f"  ❌ Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_database_stats():
    """Get database statistics"""
    db = SessionLocal()
    try:
        stats = {
            "users": db.query(User).count(),
            "tournaments": db.query(Tournament).count(),
            "ai_models": db.query(AIModel).count(),
            "battles": db.query(BattleVote).count(),
            "completed_tournaments": db.query(Tournament).filter(Tournament.status == "completed").count(),
            "active_tournaments": db.query(Tournament).filter(Tournament.status == "active").count()
        }
        return stats
    finally:
        db.close()

if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    stats = get_database_stats()
    print(f"📊 Database Stats: {stats}")
