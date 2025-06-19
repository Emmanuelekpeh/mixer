import React, { createContext, useState, useContext, useEffect } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';

const ThemeContext = createContext();

export const useTheme = () => useContext(ThemeContext);

const lightTheme = {
  background: 'var(--light-bg)',
  backgroundAlt: 'var(--light-bg-alt)',
  text: 'var(--text-dark)',
  textMuted: 'var(--text-muted-light)',
  panel: 'var(--panel-bg-light)',
  border: 'rgba(0, 0, 0, 0.1)',
  shadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
  cardBackground: 'rgba(255, 255, 255, 0.7)',
  mode: 'light'
};

const darkTheme = {
  background: 'var(--dark-bg)',
  backgroundAlt: 'var(--dark-bg-alt)',
  text: 'var(--text-light)',
  textMuted: 'var(--text-muted)',
  panel: 'var(--panel-bg)',
  border: 'rgba(255, 255, 255, 0.1)',
  shadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
  cardBackground: 'rgba(25, 25, 25, 0.7)',
  mode: 'dark'
};

export const ThemeProvider = ({ children }) => {
  // Check if user has a saved preference
  const savedTheme = localStorage.getItem('preferredTheme');
  const initialTheme = savedTheme || 'dark'; // Default to dark theme
  
  const [themeMode, setThemeMode] = useState(initialTheme);
  const theme = themeMode === 'light' ? lightTheme : darkTheme;
  
  useEffect(() => {
    // Save user's preference
    localStorage.setItem('preferredTheme', themeMode);
    
    // Update document body class for global CSS changes
    if (themeMode === 'dark') {
      document.body.classList.add('dark-theme');
      document.body.classList.remove('light-theme');
    } else {
      document.body.classList.add('light-theme');
      document.body.classList.remove('dark-theme');
    }
  }, [themeMode]);
  
  const toggleTheme = () => {
    setThemeMode(prevMode => prevMode === 'light' ? 'dark' : 'light');
  };
  
  return (
    <ThemeContext.Provider value={{ theme, themeMode, toggleTheme }}>
      <StyledThemeProvider theme={theme}>
        {children}
      </StyledThemeProvider>
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;
