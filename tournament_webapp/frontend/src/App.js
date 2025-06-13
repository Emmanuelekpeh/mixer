import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// Components
import TournamentSetup from './components/TournamentSetup';
import BattleArena from './components/BattleArena';
import TournamentResults from './components/TournamentResults';
import Leaderboard from './components/Leaderboard';
import UserProfile from './components/UserProfile';
import Navigation from './components/Navigation';

// Styles
import './App.css';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
`;

const MainContent = styled(motion.main)`
  min-height: calc(100vh - 80px);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const LoadingScreen = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  color: white;
  font-size: 24px;
`;

const LoadingSpinner = styled(motion.div)`
  width: 60px;
  height: 60px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid white;
  border-radius: 50%;
  margin-bottom: 20px;
`;

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [activeTournament, setActiveTournament] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    initializeApp();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initializeApp = async () => {
    try {
      // Check for existing user session
      const savedUser = localStorage.getItem('tournamentUser');
      if (savedUser) {
        const user = JSON.parse(savedUser);
        setCurrentUser(user);
        
        // Check for active tournament
        const activeTournamentId = localStorage.getItem('activeTournament');
        if (activeTournamentId) {
          await loadTournamentStatus(activeTournamentId);
        }
      }
      
      setLoading(false);
    } catch (err) {
      console.error('App initialization failed:', err);
      setError('Failed to initialize app');
      setLoading(false);
    }
  };

  const loadTournamentStatus = async (tournamentId) => {
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}`);
      const data = await response.json();
      
      if (data.success) {
        setActiveTournament(data.tournament);
        localStorage.setItem('activeTournament', tournamentId);
      }
    } catch (err) {
      console.error('Failed to load tournament status:', err);
      localStorage.removeItem('activeTournament');
    }
  };

  const handleUserLogin = (user) => {
    setCurrentUser(user);
    localStorage.setItem('tournamentUser', JSON.stringify(user));
  };

  const handleTournamentStart = (tournament) => {
    setActiveTournament(tournament);
    localStorage.setItem('activeTournament', tournament.tournament_id);
  };

  const handleTournamentComplete = () => {
    localStorage.removeItem('activeTournament');
    setActiveTournament(null);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setActiveTournament(null);
    localStorage.removeItem('tournamentUser');
    localStorage.removeItem('activeTournament');
  };

  if (loading) {
    return (
      <AppContainer>
        <LoadingScreen
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <LoadingSpinner
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Loading AI Tournament Engine...
          </motion.div>
        </LoadingScreen>
      </AppContainer>
    );
  }

  if (error) {
    return (
      <AppContainer>
        <LoadingScreen>
          <div style={{ color: '#ff6b6b', textAlign: 'center' }}>
            <h2>⚠️ Error</h2>
            <p>{error}</p>
            <button 
              onClick={() => window.location.reload()}
              style={{
                padding: '10px 20px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                marginTop: '20px'
              }}
            >
              Retry
            </button>
          </div>
        </LoadingScreen>
      </AppContainer>
    );
  }

  return (
    <AppContainer>
      <Router>
        <Navigation 
          user={currentUser} 
          onLogout={handleLogout}
          activeTournament={activeTournament}
        />
        
        <MainContent
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Routes>
            {/* Home / Tournament Setup */}
            <Route path="/" element={
              !currentUser ? (
                <TournamentSetup onUserLogin={handleUserLogin} />
              ) : activeTournament && activeTournament.status !== 'completed' ? (
                <Navigate to="/battle" replace />
              ) : (
                <TournamentSetup 
                  user={currentUser}
                  onTournamentStart={handleTournamentStart}
                />
              )
            } />

            {/* Battle Arena */}
            <Route path="/battle" element={
              activeTournament && activeTournament.status !== 'completed' ? (
                <BattleArena 
                  tournament={activeTournament}
                  user={currentUser}
                  onTournamentUpdate={setActiveTournament}
                  onTournamentComplete={handleTournamentComplete}
                />
              ) : (
                <Navigate to="/" replace />
              )
            } />

            {/* Tournament Results */}
            <Route path="/results/:tournamentId" element={
              <TournamentResults 
                user={currentUser}
                onNewTournament={() => setActiveTournament(null)}
              />
            } />

            {/* Leaderboard */}
            <Route path="/leaderboard" element={
              <Leaderboard user={currentUser} />
            } />

            {/* User Profile */}
            <Route path="/profile" element={
              currentUser ? (
                <UserProfile user={currentUser} />
              ) : (
                <Navigate to="/" replace />
              )
            } />

            {/* Catch all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </MainContent>

        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#333',
              color: '#fff',
              borderRadius: '8px',
            },
          }}
        />
      </Router>
    </AppContainer>
  );
}

export default App;
