import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FiUser, FiAward, FiTrendingUp, FiShare2, FiGift } from 'react-icons/fi';

const ProfileContainer = styled(motion.div)`
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 20px;
`;

const ProfileHeader = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 40px;
  text-align: center;
  margin-bottom: 30px;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const Avatar = styled.div`
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 2.5rem;
  font-weight: bold;
  margin: 0 auto 20px;
`;

const Username = styled.h1`
  color: white;
  font-size: 2.5rem;
  margin-bottom: 10px;
  font-weight: 700;
`;

const TierBadge = styled.div`
  display: inline-block;
  background: ${props => {
    switch (props.tier?.toLowerCase()) {
      case 'legend': return 'linear-gradient(135deg, #FFD700, #FFA500)';
      case 'expert': return 'linear-gradient(135deg, #C0C0C0, #808080)';
      case 'professional': return 'linear-gradient(135deg, #CD7F32, #8B4513)';
      case 'amateur': return 'linear-gradient(135deg, #4CAF50, #45a049)';
      default: return 'linear-gradient(135deg, #667eea, #764ba2)';
    }
  }};
  color: white;
  padding: 8px 20px;
  border-radius: 20px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 20px;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const StatCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  padding: 25px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const StatIcon = styled.div`
  font-size: 2rem;
  color: #667eea;
  margin-bottom: 15px;
`;

const StatValue = styled.div`
  color: white;
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 5px;
`;

const StatLabel = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const Section = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 30px;
  margin-bottom: 30px;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const SectionTitle = styled.h2`
  color: white;
  font-size: 1.5rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
`;

const ReferralCard = styled.div`
  background: rgba(102, 126, 234, 0.2);
  border: 1px solid #667eea;
  border-radius: 16px;
  padding: 25px;
  margin-bottom: 20px;
`;

const ReferralCode = styled.div`
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  padding: 15px;
  font-family: monospace;
  font-size: 1.2rem;
  color: white;
  text-align: center;
  margin: 15px 0;
  border: 1px solid rgba(255, 255, 255, 0.2);
  position: relative;
  cursor: pointer;
  
  &:hover {
    background: rgba(0, 0, 0, 0.4);
  }
`;

const CopyButton = styled.button`
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  background: #667eea;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 0.8rem;
  cursor: pointer;
  
  &:hover {
    background: #5a6bd8;
  }
`;

const AchievementsList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
`;

const AchievementBadge = styled.div`
  background: linear-gradient(135deg, #4CAF50, #45a049);
  color: white;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 5px;
`;

const LoadingState = styled.div`
  text-align: center;
  padding: 60px;
  color: rgba(255, 255, 255, 0.7);
`;

const UserProfile = ({ user }) => {
  const [userStats, setUserStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);

  useEffect(() => {    if (user) {
      fetchUserStats();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const fetchUserStats = async () => {
    try {
      const response = await fetch(`/api/users/${user.user_id}`);
      const data = await response.json();
      
      if (data.success) {
        setUserStats(data.user);
      }
    } catch (error) {
      console.error('Failed to fetch user stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const copyReferralCode = async () => {
    try {
      await navigator.clipboard.writeText(userStats.profile.referral_code);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (error) {
      console.error('Failed to copy referral code:', error);
    }
  };

  const getNextTierProgress = () => {
    const battles = userStats?.profile?.total_battles || 0;
    const tier = userStats?.profile?.tier || 'rookie';
    
    switch (tier) {
      case 'rookie':
        return { next: 'Amateur', needed: Math.max(0, 11 - battles), total: 11 };
      case 'amateur':
        return { next: 'Professional', needed: Math.max(0, 51 - battles), total: 51 };
      case 'professional':
        return { next: 'Expert', needed: Math.max(0, 201 - battles), total: 201 };
      case 'expert':
        return { next: 'Legend', needed: Math.max(0, 500 - battles), total: 500 };
      default:
        return { next: 'Max Level', needed: 0, total: 500 };
    }
  };

  if (loading) {
    return (
      <ProfileContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <LoadingState>
          <div className="loading">Loading profile...</div>
        </LoadingState>
      </ProfileContainer>
    );
  }

  if (!userStats) {
    return (
      <ProfileContainer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <LoadingState>
          <div style={{ color: '#ff6b6b' }}>Failed to load profile</div>
        </LoadingState>
      </ProfileContainer>
    );
  }

  const nextTier = getNextTierProgress();

  return (
    <ProfileContainer
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <ProfileHeader
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
      >
        <Avatar>
          {userStats.profile.username.charAt(0).toUpperCase()}
        </Avatar>
        <Username>{userStats.profile.username}</Username>
        <TierBadge tier={userStats.profile.tier}>
          {userStats.profile.tier} Mixer
        </TierBadge>
        
        {nextTier.needed > 0 && (
          <div style={{ color: 'rgba(255, 255, 255, 0.8)', marginTop: '10px' }}>
            {nextTier.needed} battles until {nextTier.next}
          </div>
        )}
      </ProfileHeader>

      <StatsGrid>
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <StatIcon><FiAward /></StatIcon>
          <StatValue>{userStats.profile.tournaments_completed}</StatValue>
          <StatLabel>Tournaments Won</StatLabel>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <StatIcon><FiTrendingUp /></StatIcon>
          <StatValue>{userStats.profile.total_battles}</StatValue>
          <StatLabel>Total Battles</StatLabel>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <StatIcon><FiShare2 /></StatIcon>
          <StatValue>{userStats.profile.friends_referred}</StatValue>
          <StatLabel>Friends Referred</StatLabel>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <StatIcon><FiGift /></StatIcon>
          <StatValue>{userStats.profile.free_mixes_earned}</StatValue>
          <StatLabel>Free Mixes Earned</StatLabel>
        </StatCard>
      </StatsGrid>

      <Section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <SectionTitle>
          <FiShare2 />
          Referral Program
        </SectionTitle>
        
        <ReferralCard>
          <div style={{ color: 'white', marginBottom: '10px' }}>
            Share your referral code and earn free mixes!
          </div>
          <div style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '0.9rem', marginBottom: '15px' }}>
            You get 5 free mixes, your friend gets 3 free mixes
          </div>
          
          <ReferralCode onClick={copyReferralCode}>
            {userStats.profile.referral_code}
            <CopyButton>
              {copySuccess ? '✓' : 'Copy'}
            </CopyButton>
          </ReferralCode>
        </ReferralCard>
      </Section>

      {userStats.profile.achievements && userStats.profile.achievements.length > 0 && (
        <Section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <SectionTitle>
            <FiAward />
            Achievements
          </SectionTitle>
          
          <AchievementsList>
            {userStats.profile.achievements.map((achievement, index) => (
              <AchievementBadge key={index}>
                🏆 {achievement}
              </AchievementBadge>
            ))}
          </AchievementsList>
        </Section>
      )}

      {userStats.preferences && (
        <Section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
        >
          <SectionTitle>
            <FiUser />
            Your Preferences
          </SectionTitle>
          
          {userStats.preferences.top_architectures && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: 'white', marginBottom: '10px' }}>Preferred Model Types:</h4>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                {Object.entries(userStats.preferences.top_architectures).map(([arch, count]) => (
                  <div key={arch} style={{
                    background: 'rgba(102, 126, 234, 0.3)',
                    padding: '5px 15px',
                    borderRadius: '15px',
                    color: 'white',
                    fontSize: '0.9rem'
                  }}>
                    {arch}: {count} votes
                  </div>
                ))}
              </div>
            </div>
          )}

          {userStats.preferences.top_genres && (
            <div>
              <h4 style={{ color: 'white', marginBottom: '10px' }}>Preferred Genres:</h4>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                {Object.entries(userStats.preferences.top_genres).map(([genre, count]) => (
                  <div key={genre} style={{
                    background: 'rgba(118, 75, 162, 0.3)',
                    padding: '5px 15px',
                    borderRadius: '15px',
                    color: 'white',
                    fontSize: '0.9rem'
                  }}>
                    {genre}: {count} votes
                  </div>
                ))}
              </div>
            </div>
          )}
        </Section>
      )}
    </ProfileContainer>
  );
};

export default UserProfile;
