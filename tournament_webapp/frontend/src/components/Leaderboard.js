import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FiAward, FiTrendingUp, FiCpu, FiZap } from 'react-icons/fi';
import { Card, Button, PageTransition } from './ui';

const LeaderboardContainer = styled(motion.div)`
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 20px;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 40px;
`;

const Title = styled.h1`
  font-size: 3rem;
  font-weight: 800;
  color: white;
  margin-bottom: 10px;
`;

const Subtitle = styled.p`
  color: rgba(255, 255, 255, 0.7);
  font-size: 1.1rem;
`;

const EnhancedLeaderboardCard = styled(Card)`
  overflow: hidden;
`;

const LeaderboardCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const LeaderboardHeader = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px 30px;
  color: white;
  display: grid;
  grid-template-columns: 60px 1fr 100px 100px 100px 100px;
  gap: 20px;
  font-weight: 600;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  
  @media (max-width: 768px) {
    grid-template-columns: 40px 1fr 80px 80px;
    gap: 10px;
    padding: 15px 20px;
    font-size: 0.8rem;
  }
`;

const ModelRow = styled(motion.div)`
  padding: 20px 30px;
  display: grid;
  grid-template-columns: 60px 1fr 100px 100px 100px 100px;
  gap: 20px;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  
  &:hover {
    background: rgba(255, 255, 255, 0.05);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    z-index: 1;
  }
  
  &:last-child {
    border-bottom: none;
  }
  
  /* Rank highlighting */
  &:nth-child(1) {
    background: linear-gradient(to right, rgba(255, 215, 0, 0.05), transparent);
  }
  
  &:nth-child(2) {
    background: linear-gradient(to right, rgba(192, 192, 192, 0.05), transparent);
  }
  
  &:nth-child(3) {
    background: linear-gradient(to right, rgba(205, 127, 50, 0.05), transparent);
  }
  
  @media (max-width: 768px) {
    grid-template-columns: 40px 1fr 80px 80px;
    gap: 10px;
    padding: 15px 20px;
  }
`;

const Rank = styled.div`
  font-size: 1.2rem;
  font-weight: 700;
  color: ${props => {
    if (props.rank === 1) return '#FFD700';
    if (props.rank === 2) return '#C0C0C0';
    if (props.rank === 3) return '#CD7F32';
    return 'white';
  }};
  text-align: center;
`;

const ModelInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const ModelAvatar = styled.div`
  width: 45px;
  height: 45px;
  border-radius: 50%;
  background: ${props => {
    // Generate unique gradient based on first letter
    const charCode = props.children ? props.children.toString().charCodeAt(0) : 65;
    const hue1 = (charCode * 7) % 360;
    const hue2 = (hue1 + 40) % 360;
    return `linear-gradient(135deg, hsl(${hue1}, 70%, 50%) 0%, hsl(${hue2}, 70%, 40%) 100%)`;
  }};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  font-size: 1rem;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  border: 2px solid rgba(255, 255, 255, 0.1);
`;

const ModelDetails = styled.div`
  flex: 1;
`;

const ModelName = styled.div`
  color: white;
  font-weight: 600;
  font-size: 1rem;
  margin-bottom: 2px;
`;

const ModelNickname = styled.div`
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.8rem;
  font-style: italic;
`;

const Stat = styled.div`
  color: white;
  font-weight: 600;
  text-align: center;
  
  .value {
    font-size: 1.1rem;
    margin-bottom: 2px;
  }
  
  .label {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
`;

const TierBadge = styled.div`
  background: ${props => {
    switch (props.tier?.toLowerCase()) {
      case 'champion': return 'linear-gradient(135deg, #FFD700, #FFA500)';
      case 'master': return 'linear-gradient(135deg, #C0C0C0, #808080)';
      case 'expert': return 'linear-gradient(135deg, #CD7F32, #8B4513)';
      case 'intermediate': return 'linear-gradient(135deg, #4CAF50, #45a049)';
      default: return 'linear-gradient(135deg, #667eea, #764ba2)';
    }
  }};
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  text-align: center;
`;

const LoadingState = styled.div`
  text-align: center;
  padding: 60px;
  color: rgba(255, 255, 255, 0.7);
`;

const ErrorState = styled.div`
  text-align: center;
  padding: 60px;
  color: #ff6b6b;
`;

const Leaderboard = ({ user }) => {
  const [leaderboard, setLeaderboard] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  const fetchLeaderboard = async () => {
    try {
      const response = await fetch('/api/leaderboard');
      const data = await response.json();
      
      if (data.success) {
        setLeaderboard(data.leaderboard);
      } else {
        throw new Error('Failed to fetch leaderboard');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  const getRankIcon = (rank) => {
    if (rank === 1) return <span style={{ fontSize: '1.4rem' }}>👑</span>;
    if (rank === 2) return <span style={{ fontSize: '1.4rem' }}>🥈</span>;
    if (rank === 3) return <span style={{ fontSize: '1.4rem' }}>🥉</span>;
    return <span>{rank}</span>;
  };

  if (loading) {
    return (
      <LeaderboardContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Header>
          <Title>🏆 Model Leaderboard</Title>
          <Subtitle>Top performing AI mixing models</Subtitle>
        </Header>
        
        <LeaderboardCard>
          <LoadingState>
            <div className="loading">Loading leaderboard...</div>
          </LoadingState>
        </LeaderboardCard>
      </LeaderboardContainer>
    );
  }
  if (error) {
    return (
      <PageTransition>
        <LeaderboardContainer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Header>
            <Title>🏆 Model Leaderboard</Title>
            <Subtitle>Top performing AI mixing models</Subtitle>
          </Header>
          
          <EnhancedLeaderboardCard>
            <ErrorState>              <div>❌ {error}</div>
              <Button
                variant="primary"
                size="medium"
                onClick={() => window.location.reload()}
                style={{ marginTop: '20px' }}
              >
                Retry
              </Button>
            </ErrorState>
          </EnhancedLeaderboardCard>
        </LeaderboardContainer>
      </PageTransition>
    );
  }
  return (
    <PageTransition>
      <LeaderboardContainer
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Header>
          <Title>🏆 Model Leaderboard</Title>
          <Subtitle>Battle-tested AI mixing champions ranked by ELO rating</Subtitle>
        </Header>

        <EnhancedLeaderboardCard
          variant="elevated"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
        <LeaderboardHeader>
          <div>Rank</div>
          <div>Model</div>
          <div className="hide-mobile">ELO</div>
          <div className="hide-mobile">Win Rate</div>
          <div>Tier</div>
          <div className="hide-mobile">Battles</div>
        </LeaderboardHeader>

        {leaderboard.map((model, index) => (
          <ModelRow
            key={model.rank}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
          >
            <Rank rank={model.rank}>
              {getRankIcon(model.rank)}
            </Rank>
              <ModelInfo>
              <ModelAvatar>
                {model.nickname ? model.nickname.charAt(0) : 'M'}
              </ModelAvatar>
              <ModelDetails>
                <ModelName>{model.name || 'Unknown'}</ModelName>
                <ModelNickname>"{model.nickname || 'Model'}"</ModelNickname>
              </ModelDetails>
            </ModelInfo>
            
            <Stat className="hide-mobile">
              <div className="value">{model.elo_rating}</div>
              <div className="label">ELO</div>
            </Stat>
            
            <Stat className="hide-mobile">
              <div className="value">{model.win_rate}%</div>
              <div className="label">Win Rate</div>
            </Stat>
            
            <TierBadge tier={model.tier}>
              {model.tier}
            </TierBadge>
            
            <Stat className="hide-mobile">
              <div className="value">{model.battles}</div>
              <div className="label">Battles</div>
            </Stat>          </ModelRow>
        ))}
      </EnhancedLeaderboardCard>
    </LeaderboardContainer>
    </PageTransition>
  );
};

export default Leaderboard;
