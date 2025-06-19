import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiUser, FiMusic, FiTrendingUp } from 'react-icons/fi';
import { GiSkullCrossedBones } from 'react-icons/gi';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

const SetupContainer = styled(motion.div)`
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
  text-align: center;
  position: relative;
  z-index: 5;
`;

const GrungeHeader = styled.div`
  margin-bottom: 20px;
  position: relative;
`;

const Logo = styled(motion.h1)`
  font-family: var(--font-title);
  font-size: 4.3rem;
  font-weight: 700;
  color: var(--primary-gold);
  margin-bottom: 5px;
  text-transform: uppercase;
  letter-spacing: 2px;
  text-shadow: 
    0 0 10px rgba(255, 215, 0, 0.4),
    2px 2px 0px rgba(0,0,0,0.8),
    4px 4px 5px rgba(0,0,0,0.4);
  position: relative;
  display: inline-block;
  line-height: 0.9;
  
  @media (max-width: 768px) {
    font-size: 3.6rem;
  }
`;

const Tagline = styled(motion.p)`
  font-family: var(--font-title);
  font-size: 2.8rem;
  font-weight: 700;
  color: var(--text-light);
  text-transform: uppercase;
  letter-spacing: 2px;
  text-shadow: 
    2px 2px 0px rgba(0,0,0,0.8),
    4px 4px 5px rgba(0,0,0,0.4);  margin-bottom: 20px;
  line-height: 0.9;
  @media (max-width: 768px) {
    font-size: 2.2rem;
  }
`;

const MetalPanel = styled(motion.div)`
  background: rgba(25, 25, 25, 0.6);
  border-radius: 12px;
  padding: 30px;
  margin-bottom: 15px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
`;

const InputGroup = styled.div`
  margin-bottom: 18px;
  text-align: left;
`;

const Label = styled.label`
  display: block;
  color: var(--text-gold);
  font-weight: 600;  margin-bottom: 6px;
  font-size: 1.2rem;
  font-family: var(--font-subtitle);
  text-transform: uppercase;
  letter-spacing: 1px;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
  
  /* Glowing effect for hip-hop/EDM vibe */
  svg {
    filter: drop-shadow(0 0 2px var(--hiphop-orange));
    color: var(--hiphop-orange);
  }
`;

const Input = styled.input`
  width: 100%;
  padding: 15px 20px;
  border: 2px solid var(--input-border);
  border-radius: 4px;
  background: var(--input-bg);
  color: var(--text-light);
  font-size: 1.1rem;
  font-family: var(--font-body);
  transition: all 0.3s ease;
  
  /* Subtle texture */
  background-image: url("https://www.transparenttextures.com/patterns/subtle-dark-vertical.png");
  background-blend-mode: overlay;
  
  &:focus {
    outline: none;
    border-color: var(--hiphop-orange);
    background: rgba(30, 30, 30, 0.8);
    box-shadow: 0 0 15px rgba(255, 69, 0, 0.2), 
                inset 0 0 10px rgba(255, 69, 0, 0.1);
  }
  
  &::placeholder {
    color: #888;
  }
`;

const DropZone = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => prop !== 'isDragActive'
})`
  border: 2px dashed ${props => props.isDragActive ? 'var(--hiphop-orange)' : 'rgba(255, 255, 255, 0.3)'};
  border-radius: 12px;
  padding: 60px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: ${props => props.isDragActive ? 'rgba(255, 69, 0, 0.15)' : 'rgba(25, 25, 25, 0.4)'};
  position: relative;
  overflow: hidden;
  box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
  
  &:hover {
    border-color: var(--hiphop-orange);
    background: rgba(255, 69, 0, 0.15);
  }
`;

const DropZoneIcon = styled(FiUpload)`
  font-size: 3.5rem;
  color: ${props => props.isDragActive ? 'var(--hiphop-orange)' : 'var(--primary-gold)'};  margin-bottom: 12px;
  filter: drop-shadow(0 0 5px rgba(255, 69, 0, 0.5));
  /* Hip-hop inspired glow effect */
  animation: ${props => props.isDragActive ? 'pulsate 1.5s infinite alternate' : 'none'};
  
  @keyframes pulsate {
    0% { opacity: 1; filter: drop-shadow(0 0 5px rgba(255, 69, 0, 0.5)); }
    100% { opacity: 0.8; filter: drop-shadow(0 0 15px rgba(255, 69, 0, 0.8)); }
  }
`;

const DropZoneText = styled.div`
  color: ${props => props.active ? 'var(--text-light)' : 'var(--text-muted)'};
  font-family: var(--font-body);
  font-size: 1.1rem;  margin-bottom: 6px;
  text-shadow: ${props => props.active ? '0 0 10px rgba(255, 69, 0, 0.5)' : 'none'};
`;

const DropZoneHint = styled.div`
  color: #999;
  font-size: 0.9rem;
`;

const FileInfo = styled.div`
  background: #222;
  border: 1px solid #444;
  border-radius: 4px;  padding: 15px;
  margin-top: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--text-dark);
  box-shadow: inset 0 0 5px rgba(0,0,0,0.3);
`;

const ButtonContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 20px;
`;

const GrungeButton = styled(motion.button)`
  font-family: var(--font-subtitle);
  text-transform: uppercase;
  font-size: 1.5rem;
  letter-spacing: 1px;
  color: var(--text-light);
  background: linear-gradient(to bottom, #FF6347, #FF4500, #D43500);
  border: none;
  border-radius: 6px;  
  padding: 15px 40px;
  position: relative;
  text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
  box-shadow: 
    inset 0 1px 0 rgba(255,255,255,0.2),
    inset 0 -1px 0 rgba(0,0,0,0.3),
    0 5px 15px rgba(0,0,0,0.4);
  transition: all 0.2s ease;
  cursor: pointer;
  overflow: hidden;
  
  /* Hip-hop inspired border effect */
  &::after {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    bottom: -1px;
    background: linear-gradient(135deg, 
      var(--hiphop-orange), 
      transparent 20%, 
      transparent 80%, 
      var(--primary-gold)
    );
    border-radius: 7px;
    z-index: -1;
    opacity: 0.8;
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.1'/%3E%3C/svg%3E");
    opacity: 0.1;
    pointer-events: none;
  }
    &:hover {
    background: linear-gradient(to bottom, var(--primary-accent-light), var(--primary-accent));
    transform: translateY(-2px);
    box-shadow: 
      inset 0 1px 0 rgba(255,255,255,0.3),
      inset 0 -1px 0 rgba(0,0,0,0.2),
      0 0 15px rgba(30, 144, 255, 0.4),
      0 6px 10px rgba(0,0,0,0.3);
  }
  
  &:active {
    transform: translateY(1px);
    background: linear-gradient(to bottom, var(--primary-accent), var(--primary-accent-dark));
    box-shadow: 
      inset 0 2px 3px rgba(0,0,0,0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    background: #ccc;
  }
`;

const FeatureCards = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 25px;
  margin-bottom: 20px;
`;

/* Feature card with genre-specific styles */
const FeatureCard = styled.div`
  position: relative;
  z-index: 1;
  text-align: center;
  background: rgba(30, 30, 30, 0.7);
  border-radius: 12px;
  padding: 25px 15px;
  box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
  }
`;

const FeatureIcon = styled.div`
  font-size: 2.5rem;
  margin-bottom: 8px;
  filter: drop-shadow(0 3px 5px rgba(0,0,0,0.3));
  
  svg {
    color: var(--text-light);
  }
`;

const FeatureTitle = styled.h3`
  font-family: var(--font-subtitle);
  margin-bottom: 6px;
  color: var(--text-light);
  font-size: 1.5rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.5px;
`;

const FeatureDescription = styled.p`
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.4;
  font-weight: 400;
  font-family: var(--font-body);
`;

const ChainDivider = styled.div`
  height: 1px;  margin: 20px 0;
  background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.2), transparent);
  position: relative;
`;

const EnhancedTournamentSetup = ({ user, onUserLogin, onTournamentStart }) => {
  const [username, setUsername] = useState(user?.username || user?.name || '');
  const [audioFile, setAudioFile] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();

  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type.includes('audio/') || file.name.endsWith('.wav') || file.name.endsWith('.mp3')) {
        setAudioFile(file);
        toast.success('Audio file added successfully!');
      } else {
        toast.error('Please upload a valid audio file');
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.aiff', '.flac']
    },
    maxFiles: 1
  });
  const handleSubmit = async (e) => {
    e.preventDefault();
      if (!username.trim()) {
      toast.error('Please enter your artist name');
      return;
    }
    
    if (!audioFile) {
      toast.error('Please upload an audio file');
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      // Create user if not exists
      let currentUser = user;
      if (!currentUser) {
        const userResponse = await fetch('/api/users/create', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: `user_${Date.now()}`,
            username: username.trim(),
          }),
        });

        const userData = await userResponse.json();
        
        if (!userData.success) {
          throw new Error('Failed to create user');
        }
        
        currentUser = userData.profile;
        if (onUserLogin) {
          onUserLogin(currentUser);
        }
        
        toast.success(`Welcome to the arena, ${username}!`);
      }
      
      // Create tournament with uploaded audio
      const formData = new FormData();
      formData.append('user_id', currentUser.user_id);
      formData.append('username', currentUser.username);
      formData.append('max_rounds', '5');
      formData.append('audio_file', audioFile);
      formData.append('audio_features', JSON.stringify({}));

      const tournamentResponse = await fetch('/api/tournaments/upload', {
        method: 'POST',
        body: formData,
      });

      const tournamentData = await tournamentResponse.json();
      
      if (!tournamentData.success) {
        throw new Error(tournamentData.message || 'Failed to create tournament');
      }
      
      // Create a tournament object from the response
      const newTournament = {
        tournament_id: tournamentData.tournament_id,
        id: tournamentData.tournament_id,
        user_id: currentUser.user_id,
        status: 'active',
        current_round: 1,
        max_rounds: 5,
        audio_file: audioFile.name,
        pairs: tournamentData.pairs || [],
        created_at: new Date().toISOString()
      };      
      toast.success('Tournament created successfully! Starting your mixing journey...');
      
      // Set the active tournament and redirect
      if (onTournamentStart) {
        onTournamentStart(newTournament);
        navigate('/battle');
      }
    } catch (error) {
      console.error('Error creating tournament:', error);
      toast.error('Failed to create tournament. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <SetupContainer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >      <GrungeHeader>
        <Logo>
          MIXTURE
        </Logo>
        <Tagline>Create. Compete. Perfect Your Sound.</Tagline>
      </GrungeHeader>

      <form onSubmit={handleSubmit}>        <MetalPanel>
          <InputGroup>
            <Label><FiUser size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> ENTER YOUR ARTIST NAME</Label>
            <Input 
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your name or artist alias"
            />
          </InputGroup>

          <InputGroup>
            <Label><FiMusic size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> UPLOAD YOUR AUDIO</Label>
            <DropZone
              {...getRootProps()}
              isDragActive={isDragActive}
            >
              <input {...getInputProps()} />
              <DropZoneIcon size={40} />
              <DropZoneText active={isDragActive}>
                {isDragActive ? 'Drop the audio here' : 'Drag & drop your audio file here'}
              </DropZoneText>
              <DropZoneHint>or click to browse your files</DropZoneHint>
            </DropZone>

            {audioFile && (
              <FileInfo>
                <span>{audioFile.name}</span>
                <span>{(audioFile.size / (1024 * 1024)).toFixed(2)} MB</span>
              </FileInfo>
            )}
          </InputGroup>
        </MetalPanel>

        <ChainDivider />

        <FeatureCards>          <FeatureCard>            <FeatureIcon>
              <GiSkullCrossedBones />
            </FeatureIcon>
            <FeatureTitle>MIX COMPARISON</FeatureTitle>
            <FeatureDescription>
              AI models create different mixes for you to choose from
            </FeatureDescription>
          </FeatureCard>
          
          <FeatureCard>
            <FeatureIcon>
              <FiTrendingUp />
            </FeatureIcon>            <FeatureTitle>MODEL EVOLUTION</FeatureTitle>
            <FeatureDescription>
              Mix improves with each round based on your choices
            </FeatureDescription>
          </FeatureCard>
          
          <FeatureCard>
            <FeatureIcon>
              <FiMusic />
            </FeatureIcon>            <FeatureTitle>VIRAL SHARING</FeatureTitle>
            <FeatureDescription>
              Share your custom mix with friends and followers
            </FeatureDescription>
          </FeatureCard>
        </FeatureCards>

        <ButtonContainer>
          <GrungeButton
            type="submit"
            disabled={isSubmitting}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isSubmitting ? 'Loading...' : 'START MIXING SESSION'}
          </GrungeButton>
        </ButtonContainer>
      </form>
    </SetupContainer>
  );
};

export default EnhancedTournamentSetup;