import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import { FiHome, FiAward, FiUser, FiLogOut, FiZap } from 'react-icons/fi';
import ThemeToggle from './ui/ThemeToggle';

const NavContainer = styled(motion.nav)`
  background: ${({ theme }) => theme.mode === 'dark' 
    ? 'rgba(0, 0, 0, 0.8)' 
    : 'rgba(255, 255, 255, 0.8)'};
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 2px solid ${({ theme }) => theme.mode === 'dark' 
    ? '#333' 
    : 'rgba(0, 0, 0, 0.1)'};
  box-shadow: 0 5px 15px ${({ theme }) => theme.mode === 'dark' 
    ? 'rgba(0, 0, 0, 0.5)' 
    : 'rgba(0, 0, 0, 0.1)'};
  padding: 15px 0;
  position: sticky;
  top: 0;
  z-index: 100;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.03;
    pointer-events: none;
    z-index: -1;
  }
`;

const NavContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
`;

const Logo = styled(Link)`
  display: flex;
  align-items: center;
  gap: 10px;  color: ${({ theme }) => theme.mode === 'dark' ? '#8B0000' : '#6B0000'};
  text-decoration: none;
  font-size: 1.7rem;
  font-family: var(--font-title);
  text-transform: uppercase;
  text-shadow: ${({ theme }) => theme.mode === 'dark' 
    ? '0 2px 4px rgba(0, 0, 0, 0.5)' 
    : '0 2px 4px rgba(0, 0, 0, 0.2)'};
  transition: all 0.3s ease;
  
  &:hover {
    transform: scale(1.03);
    color: var(--primary-red);
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
  gap: 8px;  color: ${props => props.active ? '#8B0000' : '#999'};
  text-decoration: none;
  font-weight: 500;
  font-family: var(--font-subtitle);
  font-size: 1.2rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: all 0.3s ease;
  position: relative;
  padding: 5px 10px;
  
  &:hover {
    color: #a20000;
    transform: translateY(-2px);
  }
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: ${props => props.active ? '100%' : '0'};
    height: 2px;
    background-color: #8B0000;
    transition: width 0.3s ease;
  }
  
  &:hover::after {
    width: 100%;
  }
  
  @media (max-width: 768px) {
    font-size: 1rem;
    
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
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 4px;
  padding: 5px 12px;
  box-shadow: inset 0 0 5px rgba(0,0,0,0.3);  gap: 10px;
  color: #ccc;
  
  @media (max-width: 768px) {
    gap: 5px;
  }
`;

const UserAvatar = styled.div`
  width: 35px;
  height: 35px;
  border-radius: 4px;
  background: linear-gradient(135deg, #8B0000 0%, #580000 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ddd;
  font-weight: bold;
  font-size: 0.9rem;
  border: 1px solid #444;
  text-shadow: 1px 1px 1px rgba(0,0,0,0.7);
  box-shadow: 0 2px 4px rgba(0,0,0,0.3);
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
    opacity: 0.2;
    pointer-events: none;
  }
`;

const UserName = styled.span`
  font-family: var(--font-subtitle);
  font-size: 1.2rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #aaa;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const LogoutButton = styled.button`
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 4px;
  color: #999;
  cursor: pointer;  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-subtitle);
  font-size: 1.1rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: 5px 12px;
  transition: all 0.3s ease;
  
  &:hover {
    color: #8B0000;
    border-color: #444;
    background: #222;
  }
`;

const TournamentStatus = styled.div`
  background: rgba(139, 0, 0, 0.2);
  border: 1px solid #8B0000;
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
    >      <NavContent>        <Logo to="/">
          <span>Mixture</span>
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
          
          <ThemeToggle />            {user ? (
            <>
              <UserInfo>
                <UserAvatar>
                  {user && user.username ? user.username.charAt(0).toUpperCase() : '?'}
                </UserAvatar>
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
