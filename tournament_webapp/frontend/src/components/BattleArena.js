import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { FiPlay, FiPause, FiZap } from 'react-icons/fi';
import toast from 'react-hot-toast';
import Confetti from 'react-confetti';

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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 8px 20px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 1rem;
`;

const ProgressBar = styled.div`
  width: 200px;
  height: 8px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
  
  @media (max-width: 768px) {
    width: 150px;
  }
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 4px;
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

const AudioPlayer = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 12px;
  padding: 15px;
  margin-top: 15px;
`;

const PlayButton = styled(motion.button)`
  background: ${props => props.isPlaying ? '#ff6b6b' : '#4CAF50'};
  color: white;
  border: none;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  margin: 0 auto;
  font-size: 1.2rem;
  
  &:hover {
    transform: scale(1.1);
  }
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

const VoteButton = styled(motion.button)`
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  color: white;
  border: none;
  padding: 18px 40px;
  border-radius: 12px;
  font-size: 1.2rem;
  font-weight: 600;
  cursor: pointer;
  margin-top: 20px;
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
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

const BattleArena = ({ tournament, user, onTournamentUpdate, onTournamentComplete }) => {
  const [currentBattle, setCurrentBattle] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [confidence, setConfidence] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [playingAudio, setPlayingAudio] = useState(null);
  const [showConfetti, setShowConfetti] = useState(false);
  
  const audioRefs = useRef({});
  useEffect(() => {
    if (tournament && !currentBattle) {
      startNextBattle();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tournament]);

  const startNextBattle = async () => {
    setIsLoading(true);
    setLoadingMessage('Preparing battle arena...');
    
    try {
      const response = await fetch(`/api/tournaments/${tournament.tournament_id}/battle`, {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (data.success) {
        setCurrentBattle(data.battle);
        setSelectedModel(null);
        setLoadingMessage('');
        toast.success('Battle ready! Listen to both models and vote for your favorite.');
      } else {
        throw new Error(data.error || 'Failed to start battle');
      }
    } catch (error) {
      console.error('Battle start failed:', error);
      toast.error('Failed to start battle. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVote = async () => {
    if (!selectedModel || !currentBattle) return;
    
    setIsLoading(true);
    setLoadingMessage('Recording vote and evolving models...');
    
    try {
      const response = await fetch('/api/tournaments/vote', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tournament_id: tournament.tournament_id,
          winner_id: selectedModel,
          confidence: confidence,
          reasoning: `Preferred model in round ${tournament.current_round}`
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        if (data.result.tournament_complete) {
          setShowConfetti(true);
          setTimeout(() => {
            onTournamentComplete();
            toast.success(`Tournament complete! ${data.result.champion.nickname} is your champion!`);
          }, 3000);
        } else {
          onTournamentUpdate({
            ...tournament,
            current_round: data.result.next_round,
            competitors: [data.result.winner, data.result.evolved_challenger]
          });
          setCurrentBattle(null);
          toast.success(`Round ${tournament.current_round} complete! Next challenger evolved.`);
        }
      } else {
        throw new Error(data.error || 'Failed to record vote');
      }
    } catch (error) {
      console.error('Vote recording failed:', error);
      toast.error('Failed to record vote. Please try again.');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  const playAudio = (modelId, audioPath) => {
    // Stop any currently playing audio
    Object.values(audioRefs.current).forEach(audio => {
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }
    });
    
    setPlayingAudio(null);
    
    // For demo purposes, we'll simulate audio playback
    // In production, this would play the actual audio files
    setPlayingAudio(modelId);
    
    // Simulate audio duration
    setTimeout(() => {
      setPlayingAudio(null);
    }, 30000); // 30 second preview
  };

  const stopAudio = () => {
    Object.values(audioRefs.current).forEach(audio => {
      if (audio) {
        audio.pause();
        audio.currentTime = 0;
      }
    });
    setPlayingAudio(null);
  };

  const progress = tournament.current_round / tournament.max_rounds;
  
  return (
    <ArenaContainer
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
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
      >
        <RoundInfo>
          <RoundBadge>
            Round {tournament.current_round} of {tournament.max_rounds}
          </RoundBadge>
          <ProgressBar>
            <ProgressFill
              initial={{ width: 0 }}
              animate={{ width: `${progress * 100}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            />
          </ProgressBar>
          <div style={{ color: 'rgba(255, 255, 255, 0.8)' }}>
            {Math.round(progress * 100)}% Complete
          </div>
        </RoundInfo>
      </ArenaHeader>

      {currentBattle && (
        <>
          <BattleArea
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            {/* Model A */}
            <ModelCard
              canVote={!isLoading}
              isSelected={selectedModel === currentBattle.model_a.id}
              onClick={() => !isLoading && setSelectedModel(currentBattle.model_a.id)}
              whileHover={{ scale: !isLoading ? 1.02 : 1 }}
              whileTap={{ scale: !isLoading ? 0.98 : 1 }}
            >
              <ModelAvatar>
                {currentBattle.model_a.nickname.charAt(0)}
              </ModelAvatar>
              <ModelName>{currentBattle.model_a.name}</ModelName>
              <ModelNickname>"{currentBattle.model_a.nickname}"</ModelNickname>
              
              <ModelStats>
                <Stat>
                  <div className="label">ELO</div>
                  <div className="value">{Math.round(currentBattle.model_a.elo_rating)}</div>
                </Stat>
                <Stat>
                  <div className="label">Tier</div>
                  <div className="value">{currentBattle.model_a.tier}</div>
                </Stat>
                <Stat>
                  <div className="label">Gen</div>
                  <div className="value">{currentBattle.model_a.generation}</div>
                </Stat>
              </ModelStats>
              
              <AudioPlayer>
                <PlayButton
                  isPlaying={playingAudio === currentBattle.model_a.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (playingAudio === currentBattle.model_a.id) {
                      stopAudio();
                    } else {
                      playAudio(currentBattle.model_a.id, currentBattle.audio_a);
                    }
                  }}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  {playingAudio === currentBattle.model_a.id ? <FiPause /> : <FiPlay />}
                </PlayButton>
              </AudioPlayer>
              
              {selectedModel === currentBattle.model_a.id && (
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
            </ModelCard>

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
            </VersusSection>

            {/* Model B */}
            <ModelCard
              canVote={!isLoading}
              isSelected={selectedModel === currentBattle.model_b.id}
              onClick={() => !isLoading && setSelectedModel(currentBattle.model_b.id)}
              whileHover={{ scale: !isLoading ? 1.02 : 1 }}
              whileTap={{ scale: !isLoading ? 0.98 : 1 }}
            >
              <ModelAvatar>
                {currentBattle.model_b.nickname.charAt(0)}
              </ModelAvatar>
              <ModelName>{currentBattle.model_b.name}</ModelName>
              <ModelNickname>"{currentBattle.model_b.nickname}"</ModelNickname>
              
              <ModelStats>
                <Stat>
                  <div className="label">ELO</div>
                  <div className="value">{Math.round(currentBattle.model_b.elo_rating)}</div>
                </Stat>
                <Stat>
                  <div className="label">Tier</div>
                  <div className="value">{currentBattle.model_b.tier}</div>
                </Stat>
                <Stat>
                  <div className="label">Gen</div>
                  <div className="value">{currentBattle.model_b.generation}</div>
                </Stat>
              </ModelStats>
              
              <AudioPlayer>
                <PlayButton
                  isPlaying={playingAudio === currentBattle.model_b.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (playingAudio === currentBattle.model_b.id) {
                      stopAudio();
                    } else {
                      playAudio(currentBattle.model_b.id, currentBattle.audio_b);
                    }
                  }}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  {playingAudio === currentBattle.model_b.id ? <FiPause /> : <FiPlay />}
                </PlayButton>
              </AudioPlayer>
              
              {selectedModel === currentBattle.model_b.id && (
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
            </ModelCard>
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

            <VoteButton
              onClick={handleVote}
              disabled={!selectedModel || isLoading}
              whileHover={{ scale: selectedModel && !isLoading ? 1.05 : 1 }}
              whileTap={{ scale: selectedModel && !isLoading ? 0.95 : 1 }}
            >
              {!selectedModel ? 'Select a model to vote' : 
               isLoading ? 'Processing...' : '🗳️ Cast Your Vote'}
            </VoteButton>
          </VotingSection>
        </>
      )}

      <AnimatePresence>
        {isLoading && (
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
    </ArenaContainer>
  );
};

export default BattleArena;
