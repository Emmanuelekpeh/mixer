import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import toast from 'react-hot-toast';

// Components
import EnhancedTournamentSetup from './components/EnhancedTournamentSetup';
import BattleArena from './components/BattleArena';
import TournamentResults from './components/TournamentResults';
import Leaderboard from './components/Leaderboard';
import UserProfile from './components/UserProfile';
import Navigation from './components/Navigation';
import MobileEnhancements from './components/MobileEnhancements';
import SavedTournaments from './components/SavedTournaments';
import { ThemeProvider } from './components/ThemeContext';

// Styles
import './fonts.css';
import './App.css';
import './grunge_textures.css';
import './silver_patterns.css';
import './interactive_patterns.css';
import './optimized_patterns.css';
import './intricate_patterns.css';

// Test component for dynamic tournament loading
const TestBattleArena = ({ onComplete, activeTournament }) => {
  if (!activeTournament) {
    return <div>Loading tournament...</div>;
  }
  return (
    <BattleArena 
      tournamentId={activeTournament.tournament_id}
      onComplete={onComplete}
    />
  );
};

const AppContainer = styled.div`
  min-height: 100vh;
  background-color: #dadada; /* Base background */
  background-image: 
    url("https://www.transparenttextures.com/patterns/white-paperboard.png"),
    linear-gradient(to right, rgba(130, 130, 130, 0.6) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(130, 130, 130, 0.6) 1px, transparent 1px);
  background-size: auto, 20px 20px, 20px 20px;
  font-family: var(--font-body);
  position: relative;
  overflow-x: hidden;
  
  /* Single optimized overlay that combines effects */
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    pointer-events: none;
    background-image: 
      radial-gradient(circle at 30% 30%, rgba(100, 100, 100, 0.4) 0%, transparent 50%),
      radial-gradient(circle at 70% 70%, rgba(100, 100, 100, 0.4) 0%, transparent 50%);
    transform: translateZ(0); /* Force GPU acceleration */
    animation: efficientPulse 12s infinite alternate ease-in-out;
    will-change: opacity, transform;
  }

  @keyframes efficientPulse {
    0% {
      opacity: 0.6;
      transform: translateZ(0);
    }
    100% {
      opacity: 0.8;
      transform: translateY(5px) translateZ(0);
    }
  }
  
  /* Static subtle shadow effect - no animation for better performance */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    background-image: 
      linear-gradient(45deg, transparent 65%, rgba(100, 100, 100, 0.3) 100%),
      linear-gradient(-45deg, transparent 65%, rgba(100, 100, 100, 0.3) 100%);
    opacity: 0.7;
  }

  /* Disable animations for users who prefer reduced motion */
  @media (prefers-reduced-motion: reduce) {
    &::after, &::before {
      animation: none !important;
    }
  }
  
  /* Mobile optimizations */
  @media (max-width: 768px) {
    &::after {
      animation-duration: 20s; /* Slower animations on mobile */
    }
  }
  
  &.dark-theme {
    background-color: var(--dark-bg);
    background-image: url("https://www.transparenttextures.com/patterns/black-felt.png");
  }
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
    mix-blend-mode: overlay;
    background-image: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.03) 0%, transparent 80%);
  }
`;

const MainContent = styled(motion.main)`
  min-height: calc(100vh - 80px);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  z-index: 5;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
      radial-gradient(circle at center, rgba(255, 255, 255, 0.05) 0%, transparent 80%),
      linear-gradient(to right, rgba(192, 192, 192, 0.08) 0%, transparent 50%, rgba(192, 192, 192, 0.08) 100%);
    pointer-events: none;
    z-index: -1;
    mix-blend-mode: normal;
  }
`;

// Removed unused GlassmorphicCard component

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
  const [currentUser, setCurrentUser] = useState(null);  const [activeTournament, setActiveTournament] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);  // Generate interactive procedural patterns with more intricate options - optimized for performance
  const [patternClass] = useState(() => {
    // Check if device is likely lower power
    const isLowerPowerDevice = window.innerWidth < 768 || 
                              /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Base patterns - provide the underlying structure
    // On lower power devices, use only bg-pattern-13 which is simpler
    const basePatterns = isLowerPowerDevice 
      ? ['bg-pattern-13'] 
      : ['bg-pattern-13', 'bg-pattern-11', 'bg-pattern-12', 'bg-pattern-9', 'bg-pattern-10'];
    
    // On lower power devices, use limited variations that are less CPU intensive
    const lowPowerVariations = [1, 4, 8]; // The most efficient variations
    
    // Extended variations including new intricate patterns (10-15)
    const allVariations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    
    // Select variation ID based on device capability
    const variations = isLowerPowerDevice ? lowPowerVariations : allVariations;
    const variationId = variations[Math.floor(Math.random() * variations.length)];
    
    // Always prefer bg-pattern-13 for consistency and performance
    const selectedPattern = basePatterns[Math.floor(Math.random() * basePatterns.length)];
    
    // Check for reduced motion preference - if so, use static pattern only
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      return 'bg-pattern-13 static-pattern';
    }
    
    // Combine base pattern with specific variation style
    return `${selectedPattern} pattern-variation-${variationId}`;
  });
  // Generate optimized animation properties for better performance
  const [dynamicPatternProps] = useState(() => {
    // Detect if device is likely lower power
    const isLowerPowerDevice = window.innerWidth < 768 || 
                              /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Create animation parameters based on device capability
    const animDelay = Math.random() * -5;
    const animDuration = isLowerPowerDevice ? 60 + Math.random() * 20 : 30 + Math.random() * 20; // Slower on mobile
    const opacityBase = isLowerPowerDevice ? 0.3 : 0.35; // Lower opacity on mobile
    
    // Animation intensity controls how dramatic the movements are
    const patternIntensity = isLowerPowerDevice ? 0.5 : 1; // Reduce intensity on mobile
    
    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    
    // Optimized properties
    return {
      '--pattern-anim-delay': prefersReducedMotion ? '0s' : `${animDelay}s`,
      '--pattern-anim-duration': prefersReducedMotion ? '0s' : `${animDuration}s`,
      '--pattern-opacity-base': opacityBase,
      '--pattern-intensity': prefersReducedMotion ? '0' : `${patternIntensity}`,
      
      // Start animations paused, then enable on load for better initial performance
      '--pattern-play-state': 'paused',
    };
  });
    // Enable animations after initial render for better performance
  useEffect(() => {
    // Delay animation start for better initial page load performance
    const timer = setTimeout(() => {
      document.documentElement.style.setProperty('--pattern-play-state', 'running');
    }, 300);
    
    // Optional performance monitoring
    const debugMode = false; // Set to true to enable performance monitoring
    
    if (debugMode) {
      document.body.classList.add('debug-performance');
      
      // Create performance monitoring element
      const perfMonitor = document.createElement('div');
      perfMonitor.className = 'pattern-performance-info';
      document.body.appendChild(perfMonitor);
      
      // Monitor frame rate
      let frameCount = 0;
      let lastTime = performance.now();
      
      const checkPerformance = () => {
        frameCount++;
        const now = performance.now();
        
        if (now - lastTime > 1000) {
          const fps = Math.round(frameCount * 1000 / (now - lastTime));
          const memoryInfo = window.performance?.memory ? 
            `Memory: ${Math.round(window.performance.memory.usedJSHeapSize / 1048576)}MB` : '';
          
          perfMonitor.textContent = `FPS: ${fps} ${memoryInfo}`;
          
          // If performance is poor, disable some animations
          if (fps < 30) {
            document.body.classList.add('reduce-animations');
          }
          
          frameCount = 0;
          lastTime = now;
        }
        
        requestAnimationFrame(checkPerformance);
      };
      
      requestAnimationFrame(checkPerformance);
    }
    
    // Event listeners to pause animations when tab is not visible
    const handleVisibilityChange = () => {
      if (document.hidden) {
        document.documentElement.style.setProperty('--pattern-play-state', 'paused');
      } else {
        document.documentElement.style.setProperty('--pattern-play-state', 'running');
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (debugMode) {
        const perfMonitor = document.querySelector('.pattern-performance-info');
        if (perfMonitor) {
          document.body.removeChild(perfMonitor);
        }
      }
    };
  }, []);
  
  useEffect(() => {
    initializeApp();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  const initializeApp = async () => {
    try {
      // Check for existing active tournament in localStorage
      const savedTournamentId = localStorage.getItem('activeTournament');
      
      if (savedTournamentId) {
        // Try to load the saved tournament
        try {
          console.log('🔄 Attempting to resume tournament:', savedTournamentId);
          
          // First try to resume using the resume endpoint
          const resumeResponse = await fetch(`http://localhost:10000/api/tournaments/${savedTournamentId}/resume`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          
          if (resumeResponse.ok) {
            const resumeData = await resumeResponse.json();
            if (resumeData.success && resumeData.tournament && resumeData.tournament.status !== 'completed') {
              setActiveTournament(resumeData.tournament);
              console.log('🏆 Tournament resumed successfully:', resumeData.tournament);
              toast.success('Tournament resumed! Continue where you left off.');
            } else {
              // Tournament completed, clear it
              localStorage.removeItem('activeTournament');
              console.log('� Tournament completed, cleared from storage');
            }
          } else {
            // If resume fails, try regular fetch
            console.log('🔄 Resume failed, trying regular fetch...');
            const response = await fetch(`http://localhost:10000/api/tournaments/${savedTournamentId}`);
            if (response.ok) {
              const data = await response.json();
              if (data.success && data.tournament && data.tournament.status !== 'completed') {
                setActiveTournament(data.tournament);
                console.log('🏆 Tournament loaded from regular fetch:', data.tournament);
              } else {
                // Tournament completed or invalid, clear it
                localStorage.removeItem('activeTournament');
                console.log('🏁 Tournament completed or invalid, cleared from storage');
              }
            } else {
              // Tournament not found, clear it
              localStorage.removeItem('activeTournament');
              console.log('❌ Tournament not found, cleared from storage');
            }
          }
        } catch (err) {
          console.error('Failed to resume/load saved tournament:', err);
          // Clear invalid tournament from storage
          localStorage.removeItem('activeTournament');
          toast.error('Failed to resume previous tournament. Starting fresh.');
        }
      }
        // Check for existing user session
      const savedUser = localStorage.getItem('tournamentUser');
      if (savedUser) {
        try {
          const user = JSON.parse(savedUser);
          setCurrentUser(user);
          console.log('👤 User session restored:', user.username);
        } catch (err) {
          console.error('Invalid user data in localStorage:', err);
          localStorage.removeItem('tournamentUser');
        }
      }
      // If no saved user, start with null user (will show tournament setup)
      
      setLoading(false);
    } catch (err) {
      console.error('App initialization failed:', err);
      setError('Failed to initialize app');
      setLoading(false);
    }  };

  const handleUserLogin = (user) => {
    setCurrentUser(user);
    localStorage.setItem('tournamentUser', JSON.stringify(user));
  };
  const handleTournamentStart = (tournament) => {
    setActiveTournament(tournament);
    // Handle both id and tournament_id for backend compatibility
    const tournamentId = tournament.tournament_id || tournament.id;
    localStorage.setItem('activeTournament', tournamentId);
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
    );  }  return (
    <ThemeProvider>
      <MobileEnhancements showOptimizationOverlay={true}>
        <AppContainer className={patternClass} style={dynamicPatternProps}>
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
          >          <Routes>
            {/* Home / Tournament Setup */}            <Route path="/" element={
              !currentUser ? (
                <EnhancedTournamentSetup onUserLogin={handleUserLogin} />
              ) : (
                <SavedTournaments 
                  user={currentUser}
                  onResumeTournament={handleTournamentStart}
                  onCreateNew={() => {
                    // We can't use navigate here, so we'll use window.location temporarily
                    window.location.href = '/setup';
                  }}
                />
              )
            } />

            {/* Tournament Setup */}
            <Route path="/setup" element={
              currentUser ? (
                <EnhancedTournamentSetup 
                  user={currentUser}
                  onTournamentStart={handleTournamentStart}
                />
              ) : (
                <Navigate to="/" replace />
              )
            } />            {/* Battle Arena */}
            <Route path="/battle" element={
              activeTournament && activeTournament.status !== 'completed' ? (
                <BattleArena 
                  tournamentId={activeTournament.tournament_id || activeTournament.id}
                  initialTournamentData={activeTournament}
                  onComplete={handleTournamentComplete}
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
              )            } />            {/* Test route for debugging */}
            <Route path="/test" element={
              <TestBattleArena 
                activeTournament={activeTournament}
                onComplete={handleTournamentComplete}
              />
            } />

            {/* Catch all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </MainContent>        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {              background: '#222',
              color: '#ddd',
              borderRadius: '4px',
              border: '1px solid #444',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
              fontFamily: "var(--font-body)",
              fontSize: '1.1rem',
              letterSpacing: '0.5px',
              padding: '12px 20px',
            },
            success: {
              style: {
                border: '1px solid #004400',
                background: '#111',
              },
              iconTheme: {
                primary: '#8B0000',
                secondary: '#111',
              },
            },
            error: {
              style: {
                border: '1px solid #8B0000',
                background: '#111',
              },
              iconTheme: {
                primary: '#8B0000',
                secondary: '#111',
              },
            },          }}        />
      </Router>
    </AppContainer>
      </MobileEnhancements>
    </ThemeProvider>
  );
}

export default App;
