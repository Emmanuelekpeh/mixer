import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useTheme } from '../ThemeContext';
import { FiSun, FiMoon } from 'react-icons/fi';

const ThemeToggleButton = styled(motion.button)`
  background: transparent;
  border: none;
  color: ${({ theme }) => theme.text};
  cursor: pointer;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
  }
  
  svg {
    font-size: 1.2rem;
  }
`;

const ThemeToggle = () => {
  // Get theme context safely with default values to prevent errors
  const theme = useTheme() || { themeMode: 'dark', toggleTheme: () => {} };
  const { themeMode, toggleTheme } = theme;
  
  return (
    <ThemeToggleButton
      onClick={toggleTheme}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      {themeMode === 'dark' ? <FiSun /> : <FiMoon />}
    </ThemeToggleButton>
  );
};

export default ThemeToggle;
