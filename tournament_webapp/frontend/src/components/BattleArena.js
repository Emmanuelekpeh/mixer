import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { FiZap, FiDownload } from 'react-icons/fi';
import toast from 'react-hot-toast';
import Confetti from 'react-confetti';
import { Card, Button, PageTransition } from './ui';
import BattleAudioPlayer from './BattleAudioPlayer';

const ArenaContainer = styled(motion.div)`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
`;

const ArenaHeader = styled(motion.div)`
  text-align: center;
  margin-bottom: 40px;
`;

const RoundInfo = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  margin-bottom: 20px;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 10px;
  }
`;

const RoundBadge = styled.div`
  background: linear-gradient(to bottom, #a20000, #8B0000, #580000);
  color: white;
  padding: 8px 20px;  border-radius: 4px;
  font-weight: 600;
  font-size: 1.1rem;
  text-transform: uppercase;
  font-family: var(--font-subtitle);
  letter-spacing: 1px;
  border: 1px solid #222;
  box-shadow: 
    inset 0 1px 0 rgba(255,255,255,0.1),
    0 2px 4px rgba(0,0,0,0.3);
  text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.1'/%3E%3C/svg%3E");
    opacity: 0.15;
    pointer-events: none;
  }
`;

const ProgressBar = styled.div`
  width: 200px;
  height: 12px;
  background: #111;
  border: 1px solid #333;
  border-radius: 2px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: repeating-linear-gradient(
      -45deg,
      rgba(0,0,0,0.2),
      rgba(0,0,0,0.2) 5px,
      rgba(0,0,0,0) 5px,
      rgba(0,0,0,0) 10px
    );
    pointer-events: none;
  }
  
  @media (max-width: 768px) {
    width: 150px;
  }
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #8B0000, #a20000);
  box-shadow: 0 0 10px rgba(139, 0, 0, 0.7);
  border-radius: 0;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 50%;
    background: rgba(255, 255, 255, 0.1);
  }
`;

const BattleArea = styled(motion.div)`
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 30px;
  align-items: center;
  margin-bottom: 40px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 20px;
  }
`;

const VSCircle = styled(motion.div)`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #a20000 0%, #8B0000 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.8rem;
  font-weight: bold;
  color: white;
  box-shadow: 
    0 0 20px rgba(139, 0, 0, 0.5),
    inset 0 0 10px rgba(0, 0, 0, 0.3);
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  position: relative;
  z-index: 5;
  
  &::before {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: radial-gradient(circle, rgba(139, 0, 0, 0.4) 0%, rgba(139, 0, 0, 0) 70%);
    z-index: -1;
    border-radius: 50%;
  }
  
  @media (max-width: 768px) {
    margin: 20px auto;
  }
`;

const EnhancedModelCard = styled(Card)`
  text-align: center;
  cursor: ${props => props.canVote ? 'pointer' : 'default'};
  border: ${props => props.isWinner 
    ? '3px solid #4CAF50' 
    : props.isSelected 
      ? '3px solid #667eea' 
      : '1px solid var(--glass-border)'};
  
  ${props => props.isWinner && `
    box-shadow: 0 0 15px rgba(76, 175, 80, 0.5);
  `}
  
  ${props => props.canVote && `
    &:hover {
      transform: translateY(-5px);
      border-color: #667eea;
    }
  `}
`;

// Replace the old PlayButton with our enhanced version
const PlayButton = styled(Button)`
  width: 50px;
  height: 50px;
  border-radius: 50%;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 15px auto;
  font-size: 1.2rem;
  
  svg {
    margin: 0;
  }
`;

// Replace the old VoteButton with our enhanced version
const VoteButton = styled(Button)`
  width: 100%;
  margin-top: 30px;
  font-size: 1.2rem;
  padding: 15px 0;
  font-weight: 600;
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ModelCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 30px;
  text-align: center;
  border: 3px solid ${props => props.isWinner ? '#4CAF50' : props.isSelected ? '#667eea' : 'rgba(255, 255, 255, 0.2)'};
  cursor: ${props => props.canVote ? 'pointer' : 'default'};
  position: relative;
  overflow: hidden;
  
  &:hover {
    transform: ${props => props.canVote ? 'translateY(-5px)' : 'none'};
    border-color: ${props => props.canVote ? '#667eea' : props.borderColor};
  }
  
  transition: all 0.3s ease;
`;

const ModelAvatar = styled.div`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  margin: 0 auto 20px;
  color: white;
  font-weight: bold;
`;

const ModelName = styled.h3`
  color: white;
  font-size: 1.4rem;
  margin-bottom: 10px;
  font-weight: 700;
`;

const ModelNickname = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 1rem;
  margin-bottom: 15px;
  font-style: italic;
`;

const ModelStats = styled.div`
  display: flex;
  justify-content: space-around;
  margin-bottom: 20px;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 10px;
  }
`;

const Stat = styled.div`
  text-align: center;
  
  .label {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .value {
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 2px;
  }
`;

const ModelAudioPlayer = styled.div`
  margin-top: 15px;
  /* Remove background and padding since BattleAudioPlayer handles its own styling */
`;

const VersusSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  
  @media (max-width: 768px) {
    order: -1;
  }
`;

const VersusText = styled(motion.div)`
  font-size: 3rem;
  font-weight: 900;
  color: white;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
  margin-bottom: 20px;
  
  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const VotingSection = styled(motion.div)`
  text-align: center;
  margin-bottom: 40px;
`;

const ConfidenceSlider = styled.div`
  margin: 30px 0;
`;

const SliderLabel = styled.div`
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 15px;
  font-weight: 600;
`;

const Slider = styled.input`
  width: 300px;
  height: 8px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.2);
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    border: 2px solid white;
  }
  
  &::-moz-range-thumb {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    border: 2px solid white;
  }
  
  @media (max-width: 768px) {
    width: 250px;
  }
`;

const ConfidenceValue = styled.div`
  color: white;
  font-size: 1.2rem;
  font-weight: 600;
  margin-top: 10px;
`;

const LoadingOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const LoadingSpinner = styled(motion.div)`
  width: 60px;
  height: 60px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid #667eea;
  border-radius: 50%;
  margin-bottom: 20px;
`;

const LoadingText = styled.div`
  color: white;
  font-size: 1.2rem;
  text-align: center;
`;

const DownloadButton = styled(motion.button)`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 8px;
  color: white;
  padding: 8px 12px;
  margin-top: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  justify-content: center;
  transition: all 0.3s ease;
  
  &:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  }
  
  &:active {
    transform: translateY(0);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
  
  svg {
    font-size: 1rem;
  }
`;

const BattleArena = ({ tournamentId, initialTournamentData, onComplete }) => {
  const [tournament, setTournament] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentRound, setCurrentRound] = useState(null);  const [currentPair, setCurrentPair] = useState(0);
  const [selectedModel, setSelectedModel] = useState(null);
  const [votingDisabled, setVotingDisabled] = useState(false);
  const [hasVoted, setHasVoted] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,  });
  const [showVictoryScreen, setShowVictoryScreen] = useState(false);
  const [victorModel, setVictorModel] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState('Loading tournament data...');
  const [confidence, setConfidence] = useState(0.8); // Default confidence value
  useEffect(() => {
    const fetchTournament = async () => {
      setLoading(true);
      setError(null);
        // If we have initial tournament data, use it immediately
      if (initialTournamentData) {
        console.log('🚀 Using initial tournament data:', initialTournamentData);
        console.log('🚀 Initial current_pair:', initialTournamentData.current_pair);
        console.log('🚀 Initial pairs length:', initialTournamentData.pairs?.length);
        setTournament(initialTournamentData);
        setCurrentRound(initialTournamentData.current_round);
        setCurrentPair(initialTournamentData.current_pair || 0);
        setLoading(false);
        return;
      }
      
      // Otherwise fetch from API
      try {
        const response = await fetch(`http://localhost:10000/api/tournaments/${tournamentId}`);
        const data = await response.json();
          if (data.success) {
          console.log('📊 Fetched tournament data from API:', data.tournament);
          setTournament(data.tournament);
          setCurrentRound(data.tournament.current_round);
          setCurrentPair(data.tournament.current_pair || 0);
          setLoading(false);
        } else {
          throw new Error(data.error || 'Failed to fetch tournament data');
        }
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchTournament();
  }, [tournamentId, initialTournamentData]);
  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Handle onComplete callback properly
  const handleTournamentComplete = () => {
    if (onComplete && typeof onComplete === 'function') {
      onComplete();
    }
  };
  useEffect(() => {
    if (tournament) {
      // Only update if the round actually changed
      if (currentRound !== tournament.current_round) {
        setCurrentRound(tournament.current_round);
        setCurrentPair(0); // Reset to 0 only when round changes
        setSelectedModel(null);
        setHasVoted(false);
        setVotingDisabled(false);
        setShowConfetti(false);
      } else {        // If round hasn't changed, sync currentPair with backend data
        if (tournament.current_pair !== undefined && tournament.current_pair !== currentPair) {
          console.log('🔄 Syncing currentPair from', currentPair, 'to tournament.current_pair:', tournament.current_pair);
          setCurrentPair(tournament.current_pair);
        }
      }
    }
  }, [tournament, currentRound, currentPair]);  // Additional effect to ensure currentPair stays in sync after data merging
  useEffect(() => {
    if (tournament && tournament.current_pair !== undefined && tournament.current_pair !== currentPair) {
      console.log('🔄 Emergency sync: currentPair', currentPair, '→', tournament.current_pair);
      setCurrentPair(tournament.current_pair);
      // Also reset voting state when pair changes
      setSelectedModel(null);
      setVotingDisabled(false);
      setHasVoted(false);
    }
  }, [tournament, currentPair]);

  const handleVote = async () => {
    if (!selectedModel || !tournament) return;
    
    setLoading(true);
    setError(null);
      try {      const voteData = {
        tournament_id: tournament.tournament_id,
        winner_id: selectedModel,
        confidence: 0.8,
        user_id: tournament.user_id || "test_user_audio",
        reasoning: "User selection via battle arena"
      };
      
      console.log('🗳️ Sending vote data:', voteData);
      
      const response = await fetch('http://localhost:10000/api/tournaments/vote-db', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(voteData),
      });
      
      console.log('📊 Vote response status:', response.status);
        const data = await response.json();
      console.log('📋 Vote response data:', data);
      console.log('🏆 Tournament data in response:', data.tournament);
      console.log('📊 Current pair in response:', data.tournament.current_pair);
      console.log('🎮 Total pairs:', data.tournament.pairs?.length);
        if (data.success) {
        setHasVoted(true);
        setVotingDisabled(true);
        toast.success('Your vote has been recorded!');        // Update tournament data by merging with existing data (don't overwrite!)
        if (data.tournament) {
          console.log('🔄 Merging tournament data:', data.tournament);
          console.log('🔄 Setting currentPair from', currentPair, 'to:', data.tournament.current_pair);
          console.log('🔄 Tournament status:', data.tournament.status);
          console.log('🔄 Total pairs:', data.tournament.pairs?.length || 'unknown');
          
          // Merge the response data with existing tournament data
          setTournament(prevTournament => {
            const mergedTournament = {
              ...prevTournament,  // Keep all existing data (pairs, max_rounds, etc.)
              ...data.tournament,  // Update with new status/current_pair from response
              // Ensure we keep the pairs array if it exists
              pairs: data.tournament.pairs || prevTournament?.pairs
            };
            console.log('🔄 Merged tournament:', mergedTournament);
            return mergedTournament;
          });
          
          // Update current pair state - force update even if same value
          console.log('🔄 Force updating currentPair to:', data.tournament.current_pair);
          setCurrentPair(data.tournament.current_pair);
          
          // Force a brief loading state to ensure UI updates
          setLoading(true);
          setTimeout(() => {
            setLoading(false);
            console.log('🔄 Loading state cleared, current pair should be:', data.tournament.current_pair);
          }, 100);
          
          console.log('🔄 Updated currentPair to:', data.tournament.current_pair);
          console.log('🎮 Tournament progression check: currentPair will be', data.tournament.current_pair, 'out of', data.tournament.pairs?.length || 'unknown', 'pairs');
        }
        
        // Check if tournament is complete
        if (data.tournament && data.tournament.status === 'completed') {
          setShowConfetti(true);
          setTimeout(() => {
            setShowVictoryScreen(true);
            setVictorModel(data.tournament.winner);
            if (data.tournament.winner && data.tournament.winner.nickname) {
              toast.success(`🎉 ${data.tournament.winner.nickname} has won the tournament!`);
            }
            if (onComplete && typeof onComplete === 'function') {
              onComplete();
            }
          }, 3000);
        } else {
          // Reset voting state for next pair - do this immediately, not with timeout
          setSelectedModel(null);
          setVotingDisabled(false);
          setHasVoted(false);
          console.log('🔄 Reset voting state for next pair');
        }
      } else {
        throw new Error(data.error || 'Failed to record vote');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  const handleModelSelect = (modelId) => {
    console.log('🎯 handleModelSelect called with:', modelId);
    console.log('🔒 votingDisabled:', votingDisabled);
    console.log('🎲 selectedModel before:', selectedModel);
    
    if (votingDisabled) {
      console.log('❌ Voting disabled, returning early');
      return;
    }
    
    setSelectedModel(prev => {
      const newSelection = prev === modelId ? null : modelId;
      console.log('✅ Setting selectedModel to:', newSelection);
      return newSelection;
    });  };

  // Simple stopAudio function for the AudioPlayer callback
  const stopAudio = () => {
    // This is now handled by the AudioPlayer component
    console.log('Audio stop requested');
  };  // Calculate progress safely with comprehensive null checking
  const progress = tournament?.current_round && tournament?.max_rounds ? 
    (tournament.current_round / tournament.max_rounds) : 0;
    
  // Calculate pair progress as backup
  const pairProgress = tournament?.current_pair !== undefined && tournament?.pairs?.length ? 
    (tournament.current_pair / tournament.pairs.length) : 0;
    // Debug logging
  console.log('🏆 Tournament data:', tournament);
  console.log('📊 Current pair:', currentPair);
  console.log('🎯 Selected model:', selectedModel);
  console.log('🔒 Voting disabled:', votingDisabled);
  console.log('⚡ Loading:', loading);
  if (tournament?.pairs?.[currentPair]) {
    console.log('📝 Current pair data:', tournament.pairs[currentPair]);
    console.log('🅰️ Model A ID:', tournament.pairs[currentPair]?.model_a?.id);
    console.log('🅱️ Model B ID:', tournament.pairs[currentPair]?.model_b?.id);
  } else {
    console.log('❌ No current pair data available - currentPair:', currentPair, 'pairs length:', tournament?.pairs?.length);
  }
  
  // Download mix function
  const handleDownloadMix = async (modelId, modelName, audioUrl) => {
    try {
      console.log(`🔽 Downloading mix: ${modelName} (${modelId})`);
      
      // Create download link
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      
      // Generate filename
      const tournamentPrefix = tournament?.id?.slice(-8) || 'tournament';
      const filename = `${tournamentPrefix}_${modelId}_mix.wav`;
      
      // Create and trigger download
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      toast.success(`Downloaded ${modelName} mix!`);
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download mix');    }
  };

  // Download all mixes from current battle
  const handleDownloadAll = async () => {
    try {
      const currentPairData = tournament?.pairs?.[currentPair];
      if (!currentPairData) {
        toast.error('No battle data available');
        return;
      }

      console.log('🔽 Downloading all mixes from current battle...');
      
      // Download both models' mixes
      const downloads = [
        {
          modelId: currentPairData.model_a.id,
          modelName: currentPairData.model_a.name,
          audioUrl: currentPairData.audio_a
        },
        {
          modelId: currentPairData.model_b.id,
          modelName: currentPairData.model_b.name,
          audioUrl: currentPairData.audio_b
        }
      ];

      let successCount = 0;
      
      for (const download of downloads) {
        try {
          const response = await fetch(download.audioUrl);
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          
          const tournamentPrefix = tournament?.id?.slice(-8) || 'tournament';
          const filename = `${tournamentPrefix}_${download.modelId}_mix.wav`;
          
          const link = document.createElement('a');
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
          
          successCount++;
          
          // Small delay between downloads to avoid overwhelming the browser
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
          console.error(`Failed to download ${download.modelName}:`, error);
        }
      }
      
      if (successCount === downloads.length) {
        toast.success(`Downloaded all ${successCount} mixes from Round ${tournament?.current_round || currentPair + 1}!`);
      } else if (successCount > 0) {
        toast.success(`Downloaded ${successCount} of ${downloads.length} mixes`);
      } else {
        toast.error('Failed to download any mixes');
      }
    } catch (error) {
      console.error('Download all error:', error);
      toast.error('Failed to download mixes');
    }
  };

  return (
    <PageTransition>
      <ArenaContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {showConfetti && (
          <Confetti
            width={window.innerWidth}
            height={window.innerHeight}
            recycle={false}
            numberOfPieces={500}
          />
        )}
        
        <ArenaHeader
          initial={{ y: -30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >          <RoundInfo>
            <RoundBadge>
              Round {tournament?.current_round || '?'} of {tournament?.max_rounds || '?'}
            </RoundBadge>
            <ProgressBar>
              <ProgressFill
                initial={{ width: "0%" }}
                animate={{ width: `${isNaN(progress) ? (pairProgress * 100) : (progress * 100)}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </ProgressBar>
            <div style={{ color: 'rgba(255, 255, 255, 0.8)' }}>
              {isNaN(progress) ? 
                `Pair ${(tournament?.current_pair || 0) + 1}/${tournament?.pairs?.length || '?'}` :
                `${Math.round(progress * 100)}% Complete`
              }
            </div>
          </RoundInfo>
        </ArenaHeader>        {tournament?.pairs && tournament.pairs.length > 0 && currentPair < tournament.pairs.length && (
          <div key={`battle-pair-${currentPair}`}>
            <BattleArea
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.4 }}
            >{/* Model A */}              <EnhancedModelCard
                clickable={!loading && !votingDisabled}
                canVote={!loading && !votingDisabled}
                isSelected={selectedModel === tournament?.pairs?.[currentPair]?.model_a?.id}
                onClick={() => {
                  console.log('🎯 Model A clicked!');
                  console.log('📍 Model A ID:', tournament?.pairs?.[currentPair]?.model_a?.id);
                  handleModelSelect(tournament?.pairs?.[currentPair]?.model_a?.id);
                }}
                whileHover={{ scale: !loading ? 1.02 : 1 }}
                whileTap={{ scale: !loading ? 0.98 : 1 }}
              >                <ModelAvatar>
                  {tournament?.pairs?.[currentPair]?.model_a?.nickname ? 
                    tournament.pairs[currentPair].model_a.nickname.charAt(0) : 'A'}
                </ModelAvatar>
                <ModelName>{tournament?.pairs?.[currentPair]?.model_a?.name || 'Model A'}</ModelName>
                <ModelNickname>"{tournament?.pairs?.[currentPair]?.model_a?.nickname || 'Challenger'}"</ModelNickname>
                
                <ModelStats>
                  <Stat>
                    <div className="label">ELO</div>
                    <div className="value">{Math.round(tournament?.pairs?.[currentPair]?.model_a?.elo_rating || 0)}</div>
                  </Stat>                  <Stat>
                    <div className="label">Tier</div>
                    <div className="value">{tournament?.pairs?.[currentPair]?.model_a?.tier || 'N/A'}</div>
                  </Stat>
                  <Stat>
                    <div className="label">Gen</div>
                    <div className="value">{tournament?.pairs?.[currentPair]?.model_a?.generation || '1'}</div>                  </Stat>
                </ModelStats>                  <ModelAudioPlayer>
                  <BattleAudioPlayer
                    track={{
                      title: tournament?.pairs?.[currentPair]?.model_a?.name || 'Model A Mix',
                      artist: tournament?.pairs?.[currentPair]?.model_a?.nickname || 'AI Challenger',
                      url: tournament?.pairs?.[currentPair]?.audio_a,
                      artwork: tournament?.pairs?.[currentPair]?.model_a?.nickname?.charAt(0) || 'A'
                    }}
                    onPlayStateChange={(playing, track) => {
                      if (playing) {
                        // Stop any other audio that might be playing
                        stopAudio();
                      }
                    }}                  />
                </ModelAudioPlayer>
                
                <DownloadButton
                  onClick={() => handleDownloadMix(
                    tournament?.pairs?.[currentPair]?.model_a?.id,
                    tournament?.pairs?.[currentPair]?.model_a?.name,
                    tournament?.pairs?.[currentPair]?.audio_a
                  )}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  disabled={!tournament?.pairs?.[currentPair]?.audio_a}
                >
                  <FiDownload />
                  Download Mix
                </DownloadButton>
                
                {selectedModel === tournament?.pairs?.[currentPair]?.model_a?.id && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{
                      position: 'absolute',
                      top: 10,
                      right: 10,
                      background: '#4CAF50',
                      color: 'white',
                      padding: '5px 10px',
                      borderRadius: '15px',
                      fontSize: '0.8rem',
                      fontWeight: 'bold'
                    }}
                  >
                    ✓ SELECTED
                  </motion.div>
                )}
              </EnhancedModelCard>

              {/* VS Section */}
              <VersusSection>
                <VersusText
                  animate={{ 
                    scale: [1, 1.1, 1],
                    rotate: [0, 5, -5, 0]
                  }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity,
                    repeatType: "reverse"
                  }}
                >
                  VS
                </VersusText>
                <FiZap style={{ fontSize: '2rem', color: '#667eea' }} />
              </VersusSection>              {/* Model B */}              <EnhancedModelCard
                clickable={!loading && !votingDisabled}
                canVote={!loading && !votingDisabled}
                isSelected={selectedModel === tournament?.pairs?.[currentPair]?.model_b?.id}
                onClick={() => {
                  console.log('🎯 Model B clicked!');
                  console.log('📍 Model B ID:', tournament?.pairs?.[currentPair]?.model_b?.id);
                  handleModelSelect(tournament?.pairs?.[currentPair]?.model_b?.id);
                }}                whileHover={{ scale: !loading ? 1.02 : 1 }}
                whileTap={{ scale: !loading ? 0.98 : 1 }}
              >
                <ModelAvatar>
                  {tournament?.pairs?.[currentPair]?.model_b?.nickname ? 
                    tournament.pairs[currentPair].model_b.nickname.charAt(0) : 'B'}
                </ModelAvatar><ModelName>{tournament?.pairs?.[currentPair]?.model_b?.name || 'Model B'}</ModelName>
                <ModelNickname>"{tournament?.pairs?.[currentPair]?.model_b?.nickname || 'Contender'}"</ModelNickname>
                
                <ModelStats>
                  <Stat>
                    <div className="label">ELO</div>
                    <div className="value">{Math.round(tournament?.pairs?.[currentPair]?.model_b?.elo_rating || 0)}</div>
                  </Stat>
                  <Stat>
                    <div className="label">Tier</div>
                    <div className="value">{tournament?.pairs?.[currentPair]?.model_b?.tier || 'N/A'}</div>
                  </Stat>
                  <Stat>
                    <div className="label">Gen</div>
                    <div className="value">{tournament?.pairs?.[currentPair]?.model_b?.generation || '1'}</div>
                  </Stat>                </ModelStats>                  <ModelAudioPlayer>
                  <BattleAudioPlayer
                    track={{
                      title: tournament?.pairs?.[currentPair]?.model_b?.name || 'Model B Mix',
                      artist: tournament?.pairs?.[currentPair]?.model_b?.nickname || 'AI Contender',
                      url: tournament?.pairs?.[currentPair]?.audio_b,
                      artwork: tournament?.pairs?.[currentPair]?.model_b?.nickname?.charAt(0) || 'B'
                    }}
                    onPlayStateChange={(playing, track) => {
                      if (playing) {
                        // Stop any other audio that might be playing
                        stopAudio();
                      }
                    }}                  />
                </ModelAudioPlayer>
                
                <DownloadButton
                  onClick={() => handleDownloadMix(
                    tournament?.pairs?.[currentPair]?.model_b?.id,
                    tournament?.pairs?.[currentPair]?.model_b?.name,
                    tournament?.pairs?.[currentPair]?.audio_b
                  )}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  disabled={!tournament?.pairs?.[currentPair]?.audio_b}
                >
                  <FiDownload />
                  Download Mix
                </DownloadButton>
                
                {selectedModel === tournament?.pairs?.[currentPair]?.model_b?.id && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{
                      position: 'absolute',
                      top: 10,
                      right: 10,
                      background: '#4CAF50',
                      color: 'white',
                      padding: '5px 10px',
                      borderRadius: '15px',
                      fontSize: '0.8rem',
                      fontWeight: 'bold'
                    }}
                  >
                    ✓ SELECTED
                  </motion.div>
                )}
              </EnhancedModelCard>
            </BattleArea>

            <VotingSection
              initial={{ y: 30, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              <h3 style={{ color: 'white', marginBottom: '20px' }}>
                🎧 Vote for your favorite mix!
              </h3>
              
              <ConfidenceSlider>
                <SliderLabel>
                  How confident are you in your choice?
                </SliderLabel>              <Slider
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={confidence}
                  onChange={(e) => {
                    const value = parseFloat(e.target.value);
                    setConfidence(isNaN(value) || value < 0.1 ? 0.1 : value > 1.0 ? 1.0 : value);
                  }}
                />              <ConfidenceValue>
                  {Math.round(confidence * 100)}% Confident
                </ConfidenceValue>
              </ConfidenceSlider>

              {/* Download All Button */}
              <DownloadButton
                onClick={() => handleDownloadAll()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                style={{ 
                  background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                  marginBottom: '15px'
                }}
              >
                <FiDownload />
                Download All Mixes
              </DownloadButton>

              <VoteButton
                variant="primary"
                size="large"
                onClick={handleVote}
                disabled={!selectedModel || loading}
                fullWidth={true}
                whileHover={{ scale: selectedModel && !loading ? 1.05 : 1 }}
                whileTap={{ scale: selectedModel && !loading ? 0.95 : 1 }}
              >
                {!selectedModel ? 'Select a model to vote' : 
                 loading ? 'Processing...' : '🗳️ Cast Your Vote'}              </VoteButton>
            </VotingSection>
          </div>
        )}

        <AnimatePresence>
          {loading && (
            <LoadingOverlay
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <LoadingSpinner
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              <LoadingText>{loadingMessage}</LoadingText>
            </LoadingOverlay>
          )}
        </AnimatePresence>

        {showVictoryScreen && victorModel && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.7)',
              backdropFilter: 'blur(5px)',
              zIndex: 1000,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Card
              variant="elevated"
              style={{
                zIndex: 2000,
                maxWidth: '500px',
                width: '90%',
                border: '2px solid #4CAF50',
              }}
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                style={{ textAlign: 'center' }}
              >                <h2 style={{ color: '#4CAF50', marginBottom: '20px' }}>
                  🎉 Tournament Complete!
                </h2>
                <h3 style={{ color: 'white', marginBottom: '10px' }}>
                  Champion: {victorModel?.nickname || 'Unknown'}
                </h3>
                  <Button
                  variant="primary"
                  size="large"
                  onClick={() => {
                    setShowVictoryScreen(false);
                    handleTournamentComplete();
                  }}
                >
                  View Results
                </Button>
              </motion.div>
            </Card>
          </motion.div>
        )}
      </ArenaContainer>
    </PageTransition>
  );
};

export default BattleArena;
