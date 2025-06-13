import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const ResultsContainer = styled(motion.div)`
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
  text-align: center;
`;

const Title = styled.h1`
  color: white;
  font-size: 3rem;
  margin-bottom: 20px;
`;

const Placeholder = styled.div`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 60px 40px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.2rem;
`;

const TournamentResults = ({ user, onNewTournament }) => {
  return (
    <ResultsContainer
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <Title>🏆 Tournament Results</Title>
      <Placeholder>
        Tournament results component coming soon!
        <br />
        <br />
        This will show:
        <br />
        • Final champion model
        <br />
        • Battle history and evolution tree
        <br />
        • Final mixed audio file
        <br />
        • Social sharing options
        <br />
        • Tournament replay
      </Placeholder>
    </ResultsContainer>
  );
};

export default TournamentResults;
