import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import { FiHome, FiAward, FiUser, FiLogOut, FiZap } from 'react-icons/fi';

const NavContainer = styled(motion.nav)`
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding: 15px 0;
  position: sticky;
  top: 0;
  z-index: 100;
`;

const NavContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
`;

const Logo = styled(Link)`
  display: flex;
  align-items: center;
  gap: 10px;
  color: white;
  text-decoration: none;
  font-size: 1.5rem;
  font-weight: 800;
  
  &:hover {
    color: #667eea;
  }
`;

const NavLinks = styled.div`
  display: flex;
  align-items: center;
  gap: 30px;
  
  @media (max-width: 768px) {
    gap: 15px;
  }
`;

const NavLink = styled(Link)`
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${props => props.active ? '#667eea' : 'rgba(255, 255, 255, 0.8)'};
  text-decoration: none;
  font-weight: 500;
  transition: color 0.3s ease;
  
  &:hover {
    color: #667eea;
  }
  
  @media (max-width: 768px) {
    font-size: 0.9rem;
    
    span {
      display: none;
    }
  }
`;

const UserSection = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  color: white;
  
  @media (max-width: 768px) {
    gap: 5px;
  }
`;

const UserAvatar = styled.div`
  width: 35px;
  height: 35px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  font-size: 0.9rem;
`;

const UserName = styled.span`
  font-weight: 600;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const LogoutButton = styled.button`
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.8);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.9rem;
  
  &:hover {
    color: #ff6b6b;
  }
`;

const TournamentStatus = styled.div`
  background: rgba(102, 126, 234, 0.2);
  border: 1px solid #667eea;
  border-radius: 20px;
  padding: 8px 16px;
  color: white;
  font-size: 0.8rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  
  @media (max-width: 768px) {
    padding: 6px 12px;
    font-size: 0.7rem;
  }
`;

const Navigation = ({ user, onLogout, activeTournament }) => {
  const location = useLocation();

  return (
    <NavContainer
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <NavContent>
        <Logo to="/">
          🏆 <span>AI Tournament</span>
        </Logo>

        <NavLinks>
          <NavLink 
            to="/" 
            active={location.pathname === '/' ? 1 : 0}
          >
            <FiHome />
            <span>Home</span>
          </NavLink>
          
          <NavLink 
            to="/leaderboard" 
            active={location.pathname === '/leaderboard' ? 1 : 0}
          >
            <FiAward />
            <span>Leaderboard</span>
          </NavLink>
          
          {user && (
            <NavLink 
              to="/profile" 
              active={location.pathname === '/profile' ? 1 : 0}
            >
              <FiUser />
              <span>Profile</span>
            </NavLink>
          )}
        </NavLinks>

        <UserSection>
          {activeTournament && activeTournament.status !== 'completed' && (
            <TournamentStatus>
              <FiZap />
              Round {activeTournament.current_round}/{activeTournament.max_rounds}
            </TournamentStatus>
          )}
          
          {user ? (
            <>
              <UserInfo>
                <UserAvatar>
                  {user.username.charAt(0).toUpperCase()}
                </UserAvatar>
                <UserName>{user.username}</UserName>
              </UserInfo>
              
              <LogoutButton onClick={onLogout}>
                <FiLogOut />
                <span>Logout</span>
              </LogoutButton>
            </>
          ) : (
            <div style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
              Welcome, Guest
            </div>
          )}
        </UserSection>
      </NavContent>
    </NavContainer>
  );
};

export default Navigation;
