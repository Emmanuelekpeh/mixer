import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FiAward, FiTrendingUp, FiShare2, FiDownload, FiVolume2, FiStar, FiCpu } from 'react-icons/fi';
import { useParams, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import Confetti from 'react-confetti';
import AudioPlayer from './AudioPlayer';

const ResultsContainer = styled(motion.div)`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
`;

const Header = styled(motion.div)`
  text-align: center;
  margin-bottom: 40px;
`;

const Title = styled.h1`
  font-family: var(--font-title);
  font-size: 4rem;
  font-weight: 700;
  color: var(--primary-gold);
  margin-bottom: 20px;
  text-shadow: 
    0 0 10px rgba(255, 215, 0, 0.4),
    2px 2px 0px rgba(0,0,0,0.8);
    
  @media (max-width: 768px) {
    font-size: 3rem;
  }
`;

const Subtitle = styled.p`
  font-size: 1.3rem;
  color: var(--text-light);
  opacity: 0.8;
  margin-bottom: 30px;
`;

const ChampionCard = styled(motion.div)`
  background: rgba(25, 25, 25, 0.8);
  border-radius: 20px;
  padding: 40px;
  margin-bottom: 40px;
  border: 2px solid var(--primary-gold);
  box-shadow: 
    0 0 30px rgba(255, 215, 0, 0.3),
    0 10px 20px rgba(0, 0, 0, 0.3);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(255, 215, 0, 0.1) 50%, transparent 70%);
    animation: shimmer 3s infinite;
  }
  
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

const ChampionTitle = styled.h2`
  font-size: 2.5rem;
  color: var(--primary-gold);
  text-align: center;
  margin-bottom: 20px;
  font-family: var(--font-title);
  text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
`;

const ChampionInfo = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  align-items: center;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 20px;
  }
`;

const ChampionDetails = styled.div`
  color: var(--text-light);
`;

const ChampionName = styled.h3`
  font-size: 2rem;
  color: var(--primary-gold);
  margin-bottom: 10px;
`;

const ChampionStat = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 1.1rem;
  
  svg {
    color: var(--hiphop-orange);
  }
`;

const BattleHistory = styled(motion.div)`
  background: rgba(25, 25, 25, 0.6);
  border-radius: 15px;
  padding: 30px;
  margin-bottom: 30px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const BattleHistoryTitle = styled.h3`
  font-size: 1.8rem;
  color: var(--text-gold);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const BattleItem = styled(motion.div)`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 15px 20px;
  margin-bottom: 10px;
  display: flex;
  justify-content: between;
  align-items: center;
  border-left: 4px solid ${props => props.isWin ? 'var(--success-green)' : 'var(--danger-red)'};
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const AudioSection = styled(motion.div)`
  background: rgba(25, 25, 25, 0.6);
  border-radius: 15px;
  padding: 30px;
  margin-bottom: 30px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const AudioTitle = styled.h3`
  font-size: 1.8rem;
  color: var(--text-gold);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const AudioPlayerContainer = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
`;

const SocialActions = styled(motion.div)`
  display: flex;
  gap: 20px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 30px;
`;

const ActionButton = styled(motion.button)`
  background: linear-gradient(135deg, var(--primary-gold), var(--hiphop-orange));
  border: none;
  border-radius: 10px;
  padding: 15px 25px;
  color: black;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(255, 215, 0, 0.3);
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const StatCard = styled(motion.div)`
  background: rgba(25, 25, 25, 0.6);
  border-radius: 15px;
  padding: 25px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  text-align: center;
`;

const StatValue = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--primary-gold);
  margin-bottom: 5px;
`;

const StatLabel = styled.div`
  color: var(--text-light);
  opacity: 0.8;
`;

const TournamentResults = ({ user, onNewTournament }) => {
  const { tournamentId } = useParams();
  const navigate = useNavigate();
  const [tournament, setTournament] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showConfetti, setShowConfetti] = useState(true);
  useEffect(() => {
    const loadTournamentResults = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/tournaments/${tournamentId}`);
        const data = await response.json();
        
        if (data.success) {
          setTournament(data.tournament);
        } else {
          toast.error('Failed to load tournament results');
        }
      } catch (error) {
        console.error('Error fetching tournament results:', error);
        toast.error('Failed to load tournament results');
      } finally {
        setLoading(false);
      }
    };

    if (tournamentId) {
      loadTournamentResults();
    }
      // Hide confetti after 5 seconds
    const timer = setTimeout(() => setShowConfetti(false), 5000);
    return () => clearTimeout(timer);
  }, [tournamentId]);

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: 'AI Mixing Tournament Results',
          text: `Check out my AI mixing tournament! The ${tournament?.victor_model?.name || 'champion'} created an amazing mix!`,
          url: window.location.href
        });
      } else {
        // Fallback to clipboard
        await navigator.clipboard.writeText(window.location.href);
        toast.success('Tournament link copied to clipboard!');
      }
    } catch (error) {
      console.error('Error sharing:', error);
      toast.error('Failed to share tournament');
    }
  };

  const handleDownload = () => {
    // Mock download functionality
    toast.success('Download started! (Feature coming soon)');
  };
  const handleNewTournament = () => {
    onNewTournament();
    navigate('/');
  };

  if (loading) {
    return (
      <ResultsContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div style={{ textAlign: 'center', color: 'white', fontSize: '1.5rem' }}>
          Loading tournament results...
        </div>
      </ResultsContainer>
    );
  }

  if (!tournament) {
    return (
      <ResultsContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div style={{ textAlign: 'center', color: 'white' }}>
          <h2>Tournament not found</h2>
          <ActionButton onClick={handleNewTournament}>
            Start New Tournament
          </ActionButton>
        </div>
      </ResultsContainer>
    );
  }
  const champion = tournament.victor_model || tournament.current_battle?.model_a || { name: 'Champion Model', id: 'champion' };
  const battleHistory = tournament.battle_history || [];
  const completedRounds = battleHistory.length;

  return (
    <ResultsContainer
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {showConfetti && <Confetti recycle={false} numberOfPieces={200} />}
      
      <Header>
        <Title>🏆 Tournament Complete!</Title>
        <Subtitle>Your AI mixing tournament has concluded with spectacular results</Subtitle>
      </Header>

      <ChampionCard
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <ChampionTitle>👑 Tournament Champion</ChampionTitle>
        <ChampionInfo>
          <ChampionDetails>
            <ChampionName>{champion.name}</ChampionName>
            <ChampionStat>
              <FiCpu />
              <span>Architecture: {champion.architecture || 'Advanced AI'}</span>
            </ChampionStat>
            <ChampionStat>
              <FiTrendingUp />
              <span>Performance: Exceptional</span>
            </ChampionStat>
            <ChampionStat>
              <FiStar />
              <span>Specialties: {champion.specializations?.join(', ') || 'Professional Mixing'}</span>
            </ChampionStat>
          </ChampionDetails>
          <div style={{ textAlign: 'center' }}>
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              style={{ fontSize: '4rem', marginBottom: '10px' }}
            >
              🏆
            </motion.div>
            <div style={{ color: 'var(--primary-gold)', fontSize: '1.2rem', fontWeight: '600' }}>
              Victory Achieved!
            </div>
          </div>
        </ChampionInfo>
      </ChampionCard>

      <StatsGrid>
        <StatCard
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <StatValue>{completedRounds}</StatValue>
          <StatLabel>Rounds Completed</StatLabel>
        </StatCard>
        <StatCard
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <StatValue>{battleHistory.length}</StatValue>
          <StatLabel>Battles Fought</StatLabel>
        </StatCard>
        <StatCard
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.7 }}
        >
          <StatValue>{tournament.status === 'completed' ? '100%' : '90%'}</StatValue>
          <StatLabel>Completion Rate</StatLabel>
        </StatCard>
      </StatsGrid>      <AudioSection
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <AudioTitle>
          <FiVolume2 />
          Final Mixed Audio
        </AudioTitle>
        <AudioPlayer
          tracks={[{
            title: `${champion.name} - Final Tournament Mix`,
            artist: 'AI Generated Mix',
            url: tournament.final_mix_url || '/demo-audio.mp3',
            downloadUrl: tournament.final_mix_download_url
          }]}
          compact={false}
          fullWidth={true}
          showWaveform={true}
          onPlayStateChange={(playing, track) => {
            setIsPlaying(playing);
            if (playing) {
              toast.success(`Playing: ${track.title}`);
            }
          }}
        />
      </AudioSection>

      {battleHistory.length > 0 && (
        <BattleHistory
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.9 }}
        >
          <BattleHistoryTitle>
            <FiAward />
            Battle History
          </BattleHistoryTitle>
          {battleHistory.slice(-5).map((battle, index) => (
            <BattleItem
              key={battle.battle_id || index}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 1 + index * 0.1 }}
              isWin={battle.winner_id === champion.id}
            >
              <div>
                <strong>Round {index + 1}</strong> • Winner: {battle.winner_id}
              </div>
              <div style={{ opacity: 0.7 }}>
                Confidence: {Math.round((battle.confidence || 0.7) * 100)}%
              </div>
            </BattleItem>
          ))}
        </BattleHistory>
      )}

      <SocialActions
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        <ActionButton
          onClick={handleShare}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiShare2 />
          Share Results
        </ActionButton>
        <ActionButton
          onClick={handleDownload}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiDownload />
          Download Mix
        </ActionButton>
        <ActionButton
          onClick={handleNewTournament}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          style={{ 
            background: 'linear-gradient(135deg, var(--success-green), var(--primary-gold))',
          }}
        >
          <FiAward />
          New Tournament
        </ActionButton>
      </SocialActions>
    </ResultsContainer>
  );
};

export default TournamentResults;
