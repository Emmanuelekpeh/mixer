"""
Database service layer for Tournament Webapp
Provides high-level database operations and business logic
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import uuid
import secrets
import string

from database import (
    SessionLocal, User, Tournament, AIModel, BattleVote, 
    TournamentAnalytics, get_db
)

class DatabaseService:
    """High-level database service for tournament operations"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def close(self):
        """Close database connection"""
        self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # User Operations
    def create_user(self, user_id: str, username: str, email: str = None) -> User:
        """Create a new user with error handling for concurrent access"""
        try:
            # Check if user already exists
            existing_user = self.get_user(user_id)
            if existing_user:
                return existing_user
            
            # Generate unique referral code
            referral_code = self._generate_referral_code()
            
            user = User(
                id=user_id,
                username=username,
                email=email,
                referral_code=referral_code,
                profile_data={
                    "join_date": datetime.now().isoformat(),
                    "favorite_genres": [],
                    "preferred_models": []
                }
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
            
        except Exception as e:
            # Handle integrity errors gracefully (concurrent user creation)
            self.db.rollback()
            print(f"Warning: User creation error for {user_id}: {e}")
            
            # Try to get existing user if it was created by another process
            existing_user = self.get_user(user_id)
            if existing_user:
                return existing_user
                
            # If still failing, create user with unique timestamp suffix
            import time
            unique_id = f"{user_id}_{int(time.time())}"
            referral_code = self._generate_referral_code()
            
            user = User(
                id=unique_id,
                username=f"{username}_{int(time.time())}",
                email=email,
                referral_code=referral_code,
                profile_data={
                    "join_date": datetime.now().isoformat(),
                    "favorite_genres": [],
                    "preferred_models": []
                }
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()
    
    def update_user_stats(self, user_id: str, **kwargs) -> bool:
        """Update user statistics"""
        user = self.get_user(user_id)
        if not user:
            return False
            
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        # Update tier based on total battles
        if user.total_battles >= 500:
            user.tier = "Legend"
        elif user.total_battles >= 200:
            user.tier = "Expert"
        elif user.total_battles >= 50:
            user.tier = "Professional"
        elif user.total_battles >= 10:
            user.tier = "Amateur"
        
        self.db.commit()
        return True
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences and analytics"""
        user = self.get_user(user_id)
        if not user:
            return {}
        
        # Get user's voting patterns
        votes = self.db.query(BattleVote).filter(BattleVote.user_id == user_id).all()
        
        # Analyze preferred architectures
        arch_votes = {}
        for vote in votes:
            model = self.db.query(AIModel).filter(AIModel.id == vote.winning_model_id).first()
            if model:
                arch = model.architecture
                arch_votes[arch] = arch_votes.get(arch, 0) + 1
        
        # Get tournament completion rate
        tournaments = self.db.query(Tournament).filter(Tournament.user_id == user_id).all()
        completed = sum(1 for t in tournaments if t.status == "completed")
        completion_rate = completed / len(tournaments) if tournaments else 0
        
        return {
            "top_architectures": arch_votes,
            "completion_rate": completion_rate,
            "tournaments_created": len(tournaments),
            "total_votes": len(votes),
            "average_confidence": sum(v.confidence for v in votes) / len(votes) if votes else 0.8
        }

    # Tournament Operations
    def create_tournament(self, user_id: str, max_rounds: int = 5, 
                         audio_file: str = None) -> Tournament:
        """Create a new tournament"""
        tournament_id = f"tournament_{int(datetime.now().timestamp() * 1000)}"
        
        tournament = Tournament(
            id=tournament_id,
            user_id=user_id,
            max_rounds=max_rounds,
            original_audio_file=audio_file,
            tournament_data={
                "created_at": datetime.now().isoformat(),
                "rules": "Standard elimination tournament",
                "format": "single_elimination"
            }
        )
        
        self.db.add(tournament)
        self.db.commit()
        self.db.refresh(tournament)
        return tournament
    
    def get_tournament(self, tournament_id: str) -> Optional[Tournament]:
        """Get tournament by ID"""
        return self.db.query(Tournament).filter(Tournament.id == tournament_id).first()
    
    def update_tournament(self, tournament_id: str, **kwargs) -> bool:
        """Update tournament data"""
        tournament = self.get_tournament(tournament_id)
        if not tournament:
            return False
            
        for key, value in kwargs.items():
            if hasattr(tournament, key):
                setattr(tournament, key, value)
        
        tournament.updated_at = datetime.now()
        self.db.commit()
        return True
    
    def complete_tournament(self, tournament_id: str, victor_model_id: str) -> bool:
        """Mark tournament as completed"""
        tournament = self.get_tournament(tournament_id)
        if not tournament:
            return False
        
        tournament.status = "completed"
        tournament.victor_model_id = victor_model_id
        tournament.completed_at = datetime.now()
        tournament.completion_rate = 1.0
        
        # Update user stats
        user = self.get_user(tournament.user_id)
        if user:
            user.tournaments_completed += 1
            if victor_model_id:  # Tournament finished successfully
                user.tournaments_won += 1
        
        self.db.commit()
        return True
    
    def get_user_tournaments(self, user_id: str, limit: int = 10) -> List[Tournament]:
        """Get user's tournaments"""
        return (self.db.query(Tournament)
                .filter(Tournament.user_id == user_id)
                .order_by(desc(Tournament.created_at))
                .limit(limit)
                .all())

    # AI Model Operations
    def get_all_models(self) -> List[AIModel]:
        """Get all available AI models"""
        return self.db.query(AIModel).filter(AIModel.is_active == True).all()
    
    def get_all_models_cached(self) -> List[AIModel]:
        """Get all models with caching for performance"""
        # Simple in-memory cache to avoid repeated DB queries
        if not hasattr(self, '_models_cache') or not self._models_cache:
            self._models_cache = self.db.query(AIModel).all()
        return self._models_cache
    
    def clear_models_cache(self):
        """Clear the models cache"""
        if hasattr(self, '_models_cache'):
            self._models_cache = None
    
    def get_model(self, model_id: str) -> Optional[AIModel]:
        """Get AI model by ID"""
        return self.db.query(AIModel).filter(AIModel.id == model_id).first()
    
    def update_model_stats(self, model_id: str, opponent_id: str, 
                          won: bool, confidence: float = 0.8) -> bool:
        """Update model statistics after a battle"""
        model = self.get_model(model_id)
        opponent = self.get_model(opponent_id)
        
        if not model or not opponent:
            return False
        
        # Update battle counts
        model.total_battles += 1
        opponent.total_battles += 1
        
        # Update wins/losses
        if won:
            model.wins += 1
            opponent.losses += 1
        else:
            model.losses += 1
            opponent.wins += 1
        
        # Update win rates
        model.win_rate = model.wins / model.total_battles if model.total_battles > 0 else 0
        opponent.win_rate = opponent.wins / opponent.total_battles if opponent.total_battles > 0 else 0
        
        # Update ELO ratings
        self._update_elo_ratings(model, opponent, won, confidence)
        
        # Update last used
        model.last_used = datetime.now()
        opponent.last_used = datetime.now()
        
        self.db.commit()
        return True
    
    def get_top_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing models"""
        models = (self.db.query(AIModel)
                 .filter(AIModel.is_active == True)
                 .filter(AIModel.total_battles >= 5)
                 .order_by(desc(AIModel.elo_rating))
                 .limit(limit)
                 .all())
        
        return [{
            "id": model.id,
            "name": model.name,
            "architecture": model.architecture,
            "elo_rating": model.elo_rating,
            "win_rate": model.win_rate,
            "total_battles": model.total_battles,
            "tier": model.tier
        } for model in models]

    # Battle Vote Operations
    def record_vote(self, tournament_id: str, user_id: str, 
                   model_a_id: str, model_b_id: str, winning_model_id: str,
                   confidence: float = 0.8, round_number: int = 1,
                   pair_number: int = 0) -> BattleVote:
        """Record a battle vote"""
        vote = BattleVote(
            tournament_id=tournament_id,
            user_id=user_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            winning_model_id=winning_model_id,
            confidence=confidence,
            round_number=round_number,
            pair_number=pair_number,
            vote_data={
                "timestamp": datetime.now().isoformat(),
                "round": round_number,
                "pair": pair_number
            }
        )
        
        self.db.add(vote)
        
        # Update user stats
        user = self.get_user(user_id)
        if user:
            user.total_votes += 1
            user.total_battles += 1
        
        # Update model stats
        self.update_model_stats(
            winning_model_id, 
            model_b_id if winning_model_id == model_a_id else model_a_id,
            True, 
            confidence
        )
        
        self.db.commit()
        self.db.refresh(vote)
        return vote
    
    def get_tournament_votes(self, tournament_id: str) -> List[BattleVote]:
        """Get all votes for a tournament"""
        return (self.db.query(BattleVote)
                .filter(BattleVote.tournament_id == tournament_id)
                .order_by(BattleVote.created_at)
                .all())

    # Analytics Operations
    def get_tournament_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get tournament analytics for the last N days"""
        since_date = datetime.now() - timedelta(days=days)
        
        # Tournament stats
        tournaments = self.db.query(Tournament).filter(Tournament.created_at >= since_date).all()
        completed_tournaments = [t for t in tournaments if t.status == "completed"]
        
        # Vote stats
        votes = self.db.query(BattleVote).filter(BattleVote.created_at >= since_date).all()
        
        # User stats
        active_users = (self.db.query(User)
                       .join(Tournament)
                       .filter(Tournament.created_at >= since_date)
                       .distinct()
                       .count())
        
        # Popular models
        model_votes = {}
        for vote in votes:
            model_id = vote.winning_model_id
            model_votes[model_id] = model_votes.get(model_id, 0) + 1
        
        top_models = sorted(model_votes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "period_days": days,
            "tournaments_created": len(tournaments),
            "tournaments_completed": len(completed_tournaments),
            "completion_rate": len(completed_tournaments) / len(tournaments) if tournaments else 0,
            "total_votes": len(votes),
            "active_users": active_users,
            "top_models": top_models,
            "average_confidence": sum(v.confidence for v in votes) / len(votes) if votes else 0.8
        }

    # Helper Methods
    def _generate_referral_code(self) -> str:
        """Generate unique referral code"""
        while True:
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            existing = self.db.query(User).filter(User.referral_code == code).first()
            if not existing:
                return code
    
    def _update_elo_ratings(self, model: AIModel, opponent: AIModel, 
                           model_won: bool, confidence: float):
        """Update ELO ratings based on battle result"""
        K = 32  # K-factor for ELO calculation
        
        # Expected scores
        expected_model = 1 / (1 + 10 ** ((opponent.elo_rating - model.elo_rating) / 400))
        expected_opponent = 1 / (1 + 10 ** ((model.elo_rating - opponent.elo_rating) / 400))
        
        # Actual scores (adjusted by confidence)
        if model_won:
            actual_model = confidence
            actual_opponent = 1 - confidence
        else:
            actual_model = 1 - confidence
            actual_opponent = confidence
        
        # Update ratings
        model.elo_rating += K * (actual_model - expected_model)
        opponent.elo_rating += K * (actual_opponent - expected_opponent)
        
        # Ensure ratings don't go below 100
        model.elo_rating = max(100, model.elo_rating)
        opponent.elo_rating = max(100, opponent.elo_rating)

# Convenience functions for use in API
def get_database_service():
    """Get a database service instance - for dependency injection"""
    service = DatabaseService()
    try:
        yield service
    finally:
        service.close()

def with_db_service(func):
    """Decorator to provide database service to functions"""
    def wrapper(*args, **kwargs):
        with DatabaseService() as db_service:
            return func(db_service, *args, **kwargs)
    return wrapper
