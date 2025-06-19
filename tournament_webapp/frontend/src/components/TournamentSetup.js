import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiUser, FiMusic, FiTrendingUp, FiPlus, FiCheck, FiSettings } from 'react-icons/fi';
import toast from 'react-hot-toast';
import { Card, Button, InputField, PageTransition } from './ui';

const SetupContainer = styled(motion.div)`
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
  text-align: center;
`;

const Title = styled(motion.h1)`
  font-family: var(--font-title);
  font-size: 4rem;
  font-weight: 700;
  color: #8B0000;
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 2px;
  text-shadow: 
    2px 2px 0px rgba(0,0,0,0.8),
    4px 4px 5px rgba(0,0,0,0.4);
  position: relative;
  display: inline-block;
  
  @media (max-width: 768px) {
    font-size: 2.8rem;
  }
`;

const Subtitle = styled(motion.p)`
  font-family: var(--font-subtitle);
  font-size: 1.3rem;
  color: #aaa;
  margin-bottom: 50px;
  line-height: 1.6;
  text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
`;

// Enhanced version of SetupCard using our Card component
const EnhancedSetupCard = styled(Card)`
  margin-bottom: 30px;
  text-align: left;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.08;
    pointer-events: none;
    z-index: -1;
  }
`;

// Keep the original for backward compatibility
const SetupCard = styled(motion.div)`
  background: #111111;
  border-radius: 4px;
  padding: 30px;
  margin-bottom: 30px;
  border: 2px solid #333;
  box-shadow: 
    inset 0 0 0 1px rgba(255,255,255,0.05),
    0 10px 20px rgba(0,0,0,0.4);
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.08;
    pointer-events: none;
  }
`;

const InputGroup = styled.div`
  margin-bottom: 30px;
  text-align: left;
`;

// Original components kept for backward compatibility
const Label = styled.label`
  display: block;  color: #ccc;
  font-weight: 600;
  margin-bottom: 10px;
  font-size: 1.2rem;
  font-family: var(--font-subtitle);
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const Input = styled.input`
  width: 100%;
  padding: 15px 20px;
  border: 2px solid #333;
  border-radius: 4px;
  background: #222;  color: #ddd;
  font-size: 1.1rem;
  font-family: var(--font-body);
  transition: all 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #8B0000;
    background: #2a2a2a;
    box-shadow: 0 0 10px rgba(139, 0, 0, 0.3);
  }
`;

const DropZone = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => prop !== 'isDragActive'
})`
  border: 3px dashed ${props => props.isDragActive ? '#8B0000' : '#444'};
  border-radius: 4px;
  padding: 60px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${props => props.isDragActive ? 'rgba(139, 0, 0, 0.1)' : '#1a1a1a'};
  position: relative;
  overflow: hidden;
  
  &:hover {
    border-color: #8B0000;
    background: rgba(139, 0, 0, 0.1);
  }
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.05;
    pointer-events: none;
  }
`;

const DropZoneIcon = styled(FiUpload)`
  font-size: 3rem;
  color: ${props => props.isDragActive ? '#8B0000' : '#666'};
  margin-bottom: 20px;
  filter: drop-shadow(0 2px 3px rgba(0,0,0,0.3));
`;

const DropZoneText = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.1rem;
  margin-bottom: 10px;
`;

const DropZoneSubtext = styled.div`
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.9rem;
`;

const FileInfo = styled(motion.div)`
  background: rgba(102, 126, 234, 0.2);
  border-radius: 12px;
  padding: 20px;  margin-top: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

// Renamed to avoid conflict with imported Button
const StyledButton = styled(motion.button)`
  font-family: var(--font-subtitle);
  text-transform: uppercase;
  font-size: 1.5rem;
  letter-spacing: 1px;
  color: #fff;
  background: linear-gradient(to bottom, #a20000, #8B0000, #580000);
  border: 1px solid #222;
  border-radius: 4px;
  padding: 12px 40px;
  position: relative;
  text-shadow: 1px 1px 1px rgba(0,0,0,0.7);
  box-shadow: 
    inset 0 1px 0 rgba(255,255,255,0.1),
    inset 0 -1px 0 rgba(0,0,0,0.3),
    0 3px 5px rgba(0,0,0,0.3);
  transition: all 0.2s ease;
  cursor: pointer;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.1'/%3E%3C/svg%3E");
    opacity: 0.2;
    pointer-events: none;
  }
  
  &:hover {
    background: linear-gradient(to bottom, #cf0000, #a20000, #8B0000);
    transform: translateY(-2px);
    box-shadow: 
      inset 0 1px 0 rgba(255,255,255,0.1),
      inset 0 -1px 0 rgba(0,0,0,0.3),
      0 6px 10px rgba(0,0,0,0.5);
  }
  
  &:active {
    transform: translateY(1px);
    background: linear-gradient(to bottom, #8B0000, #580000);
    box-shadow: 
      inset 0 2px 3px rgba(0,0,0,0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    background: #333;
  }
`;

const FeatureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 30px;
  margin-top: 50px;
`;

const FeatureCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  padding: 30px 20px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const FeatureIcon = styled.div`
  font-size: 2.5rem;
  color: #667eea;
  margin-bottom: 20px;
`;

const FeatureTitle = styled.h3`
  color: white;
  font-size: 1.3rem;
  margin-bottom: 15px;
  font-weight: 600;
`;

const FeatureDescription = styled.p`
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.95rem;
  line-height: 1.5;
`;

const TournamentSetup = ({ user, onUserLogin, onTournamentStart }) => {
  const [username, setUsername] = useState(user?.username || '');
  const [audioFile, setAudioFile] = useState(null);
  const [maxRounds, setMaxRounds] = useState(5);
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      // Validate file type and size
      const validTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/aiff'];
      const maxSize = 50 * 1024 * 1024; // 50MB
      
      if (!validTypes.includes(file.type)) {
        toast.error('Please upload a valid audio file (WAV, MP3, FLAC, or AIFF)');
        return;
      }
      
      if (file.size > maxSize) {
        toast.error('File size must be less than 50MB');
        return;
      }
      
      setAudioFile(file);
      toast.success('Audio file uploaded successfully!');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.aiff']
    },
    multiple: false
  });

  const handleCreateUser = async () => {
    if (!username.trim()) {
      toast.error('Please enter a username');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/users/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: `user_${Date.now()}`,
          username: username.trim(),
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        onUserLogin(data.profile);
        toast.success(`Welcome to the arena, ${username}!`);
      } else {
        throw new Error('Failed to create user');
      }
    } catch (error) {
      console.error('User creation failed:', error);
      toast.error('Failed to create user. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };  const handleStartTournament = async () => {
    if (!audioFile) {
      toast.error('Please upload an audio file to start a tournament');
      return;
    }
    
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('user_id', user.user_id);
      formData.append('username', user.username);
      formData.append('max_rounds', maxRounds.toString());
      formData.append('audio_file', audioFile);
      formData.append('audio_features', JSON.stringify({}));

      const response = await fetch('/api/tournaments/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        const tournament = data.tournament || {
          id: data.tournament_id,
          user_id: user.user_id,
          username: user.username,
          max_rounds: maxRounds,
          pairs: data.pairs || []
        };
        
        onTournamentStart(tournament);
        toast.success('Tournament started with your uploaded audio! Let the battles begin!');
      } else {
        throw new Error(data.message || 'Failed to create tournament');
      }
    } catch (error) {
      console.error('Tournament creation failed:', error);
      toast.error('Failed to start tournament. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  if (!user) {
    return (
      <PageTransition>
        <SetupContainer
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >          <Title
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            🏆 Mixture Tournament
          </Title>
        
        <Subtitle
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          Battle AI models, evolve the losers, and create the perfect mix!
        </Subtitle>

        <EnhancedSetupCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
        >          <InputGroup>
            <Label>
              <FiUser style={{ marginRight: '8px', verticalAlign: 'middle' }} />
              Choose your fighter name
            </Label>
            <InputField
              type="text"
              placeholder="Enter your username..."
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleCreateUser()}
            />
          </InputGroup>

          <Button
            variant="primary"
            size="large"
            onClick={handleCreateUser}
            disabled={isLoading || !username.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLoading ? 'Creating Profile...' : 'Enter the Arena'}
          </Button>
        </EnhancedSetupCard>

        <FeatureGrid>
          <FeatureCard
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <FeatureIcon>
              <FiMusic />
            </FeatureIcon>
            <FeatureTitle>AI Model Battles</FeatureTitle>
            <FeatureDescription>
              Watch AI models compete head-to-head to create the best mix of your music
            </FeatureDescription>
          </FeatureCard>

          <FeatureCard
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
          >
            <FeatureIcon>
              <FiTrendingUp />
            </FeatureIcon>
            <FeatureTitle>Model Evolution</FeatureTitle>
            <FeatureDescription>
              Losing models learn from winners and evolve, creating stronger challengers
            </FeatureDescription>
          </FeatureCard>

          <FeatureCard
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
          >
            <FeatureIcon>
              🧬
            </FeatureIcon>
            <FeatureTitle>Viral Sharing</FeatureTitle>
            <FeatureDescription>
              Share your tournament results and earn free mixes for your friends
            </FeatureDescription>          </FeatureCard>
        </FeatureGrid>
      </SetupContainer>
      </PageTransition>
    );
  }

  return (
    <PageTransition>
      <SetupContainer>
        <Title
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          🎵 Upload Your Track
        </Title>
        <Subtitle
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Ready for battle, {user.username}! Upload your audio file or start a demo tournament.
        </Subtitle>

        <EnhancedSetupCard
          variant="elevated"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
        >
          <InputGroup>
            <Label>Audio File</Label>
            <DropZone
              {...getRootProps()}
              isDragActive={isDragActive}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <input {...getInputProps()} />
              <DropZoneIcon />
              <DropZoneText>
                {isDragActive ? 'Drop your audio file here!' : 'Drag & drop your audio file here'}
              </DropZoneText>              <DropZoneSubtext>
                Supports WAV, MP3, FLAC, AIFF (max 50MB) • Or skip for demo mode
              </DropZoneSubtext>
            </DropZone>

            {audioFile && (
              <FileInfo
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div>
                  <strong>{audioFile.name}</strong>
                  <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>
                    {(audioFile.size / (1024 * 1024)).toFixed(2)} MB
                  </div>                </div>
                <Button
                  variant="secondary"
                  size="small"
                  onClick={() => setAudioFile(null)}
                  style={{ padding: '5px 10px', minWidth: 'auto' }}
                >
                  ✕
                </Button>
              </FileInfo>
            )}
          </InputGroup>        <InputGroup>
          <Label>Tournament Rounds</Label>
            <InputField
              type="number"
              min="3"
              max="10"
              value={maxRounds}
              onChange={(e) => {
                const value = parseInt(e.target.value, 10);
                setMaxRounds(isNaN(value) || value < 3 ? 3 : value > 10 ? 10 : value);
              }}
            />
        </InputGroup>        <Button
          variant="primary"
          size="large"
          onClick={handleStartTournament}
          disabled={isLoading || !audioFile}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isLoading ? 'Starting Tournament...' : '🏆 Start Tournament'}
        </Button>
      </EnhancedSetupCard>
    </SetupContainer>
    </PageTransition>
  );
};

export default TournamentSetup;
