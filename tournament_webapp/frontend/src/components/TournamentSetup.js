import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiUser, FiMusic, FiTrendingUp } from 'react-icons/fi';
import toast from 'react-hot-toast';

const SetupContainer = styled(motion.div)`
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
  text-align: center;
`;

const Title = styled(motion.h1)`
  font-size: 3.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 20px;
  
  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const Subtitle = styled(motion.p)`
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 50px;
  line-height: 1.6;
`;

const SetupCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 40px;
  margin-bottom: 30px;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const InputGroup = styled.div`
  margin-bottom: 30px;
  text-align: left;
`;

const Label = styled.label`
  display: block;
  color: rgba(255, 255, 255, 0.9);
  font-weight: 600;
  margin-bottom: 10px;
  font-size: 1rem;
`;

const Input = styled.input`
  width: 100%;
  padding: 15px 20px;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 1rem;
  transition: all 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: #667eea;
    background: rgba(255, 255, 255, 0.15);
  }
  
  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }
`;

const DropZone = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => prop !== 'isDragActive'
})`
  border: 3px dashed ${props => props.isDragActive ? '#667eea' : 'rgba(255, 255, 255, 0.3)'};
  border-radius: 16px;
  padding: 60px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${props => props.isDragActive ? 'rgba(102, 126, 234, 0.1)' : 'rgba(255, 255, 255, 0.05)'};
  
  &:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.1);
  }
`;

const DropZoneIcon = styled(FiUpload)`
  font-size: 3rem;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 20px;
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
  padding: 20px;
  margin-top: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Button = styled(motion.button)`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 18px 40px;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
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
  };

  const handleStartTournament = async () => {
    if (!audioFile) {
      toast.error('Please upload an audio file');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio_file', audioFile);
      
      const requestData = {
        user_id: user.user_id,
        username: user.username,
        max_rounds: maxRounds,
        audio_features: {} // Could add audio analysis here
      };

      // Convert request data to form fields
      Object.entries(requestData).forEach(([key, value]) => {
        formData.append(key, typeof value === 'object' ? JSON.stringify(value) : value);
      });

      const response = await fetch('/api/tournaments/create', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        onTournamentStart(data.tournament);
        toast.success('Tournament started! Let the battles begin!');
      } else {
        throw new Error('Failed to create tournament');
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
      <SetupContainer
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Title
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          🏆 AI Mixer Tournament
        </Title>
        
        <Subtitle
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          Battle AI models, evolve the losers, and create the perfect mix!
        </Subtitle>

        <SetupCard
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
        >
          <InputGroup>
            <Label>
              <FiUser style={{ marginRight: '8px', verticalAlign: 'middle' }} />
              Choose your fighter name
            </Label>
            <Input
              type="text"
              placeholder="Enter your username..."
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleCreateUser()}
            />
          </InputGroup>

          <Button
            onClick={handleCreateUser}
            disabled={isLoading || !username.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLoading ? 'Creating Profile...' : 'Enter the Arena'}
          </Button>
        </SetupCard>

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
            </FeatureDescription>
          </FeatureCard>
        </FeatureGrid>
      </SetupContainer>
    );
  }

  return (
    <SetupContainer
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <Title>🎵 Upload Your Track</Title>
      <Subtitle>
        Ready for battle, {user.username}! Upload your audio file to start the tournament.
      </Subtitle>

      <SetupCard
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
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
            </DropZoneText>
            <DropZoneSubtext>
              Supports WAV, MP3, FLAC, AIFF (max 50MB)
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
                </div>
              </div>
              <button
                onClick={() => setAudioFile(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'rgba(255, 255, 255, 0.7)',
                  cursor: 'pointer',
                  fontSize: '1.2rem'
                }}
              >
                ✕
              </button>
            </FileInfo>
          )}
        </InputGroup>        <InputGroup>
          <Label>Tournament Rounds</Label>            <Input
              type="number"
              min="3"
              max="10"
              value={maxRounds}
              onChange={(e) => {
                const value = parseInt(e.target.value, 10);
                setMaxRounds(isNaN(value) || value < 3 ? 3 : value > 10 ? 10 : value);
              }}
            />
        </InputGroup>

        <Button
          onClick={handleStartTournament}
          disabled={isLoading || !audioFile}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isLoading ? 'Starting Tournament...' : '🏆 Start Tournament'}
        </Button>
      </SetupCard>
    </SetupContainer>
  );
};

export default TournamentSetup;
