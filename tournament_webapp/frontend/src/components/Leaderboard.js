import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

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
  transition: background 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.05);
  }
  
  &:last-child {
    border-bottom: none;
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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  font-size: 1rem;
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
    if (rank === 1) return '👑';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return rank;
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
      <LeaderboardContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Header>
          <Title>🏆 Model Leaderboard</Title>
          <Subtitle>Top performing AI mixing models</Subtitle>
        </Header>
        
        <LeaderboardCard>
          <ErrorState>
            <div>❌ {error}</div>
            <button 
              onClick={() => window.location.reload()}
              style={{
                marginTop: '20px',
                padding: '10px 20px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          </ErrorState>
        </LeaderboardCard>
      </LeaderboardContainer>
    );
  }

  return (
    <LeaderboardContainer
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <Header>
        <Title>🏆 Model Leaderboard</Title>
        <Subtitle>Battle-tested AI mixing champions ranked by ELO rating</Subtitle>
      </Header>

      <LeaderboardCard
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
                {model.nickname.charAt(0)}
              </ModelAvatar>
              <ModelDetails>
                <ModelName>{model.name}</ModelName>
                <ModelNickname>"{model.nickname}"</ModelNickname>
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
            </Stat>
          </ModelRow>
        ))}
      </LeaderboardCard>
    </LeaderboardContainer>
  );
};

export default Leaderboard;
