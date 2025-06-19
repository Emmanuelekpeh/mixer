import React, { useState, useRef, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  FiPlay, 
  FiPause, 
  FiSkipBack, 
  FiSkipForward,
  FiVolume2, 
  FiVolumeX 
} from 'react-icons/fi';
import toast from 'react-hot-toast';

const BattlePlayerContainer = styled(motion.div)`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(15px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  padding: 20px;
  margin-top: 20px;
  box-shadow: 
    0 8px 25px rgba(0, 0, 0, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.15);
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(102, 126, 234, 0.3);
    box-shadow: 
      0 12px 35px rgba(0, 0, 0, 0.2),
      0 0 0 1px rgba(102, 126, 234, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
  }
`;

const PlayerHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
`;

const TrackArtwork = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
  color: white;
  flex-shrink: 0;
  box-shadow: 
    0 4px 12px rgba(102, 126, 234, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.3);
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
    border-radius: 14px;
    z-index: -1;
  }
`;

const TrackInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const TrackTitle = styled.div`
  color: white;
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const TrackArtist = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.8rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ProgressSection = styled.div`
  margin-bottom: 12px;
`;

const ProgressBar = styled.div`
  height: 6px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
  position: relative;
  cursor: pointer;
  margin-bottom: 10px;
  overflow: hidden;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
    height: 8px;
    margin-bottom: 9px;
  }
  
  transition: all 0.2s ease;
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 3px;
  position: relative;
  box-shadow: 0 0 8px rgba(102, 126, 234, 0.5);
  
  &::after {
    content: '';
    position: absolute;
    right: -6px;
    top: 50%;
    transform: translateY(-50%);
    width: 12px;
    height: 12px;
    background: white;
    border-radius: 50%;
    box-shadow: 
      0 2px 6px rgba(0, 0, 0, 0.3),
      0 0 0 2px rgba(102, 126, 234, 0.5);
    transition: all 0.2s ease;
  }
  
  &:hover::after {
    transform: translateY(-50%) scale(1.2);
  }
`;

const TimeDisplay = styled.div`
  display: flex;
  justify-content: space-between;
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.7rem;
  font-family: monospace;
`;

const ControlsSection = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const MainControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ControlButton = styled(motion.button).withConfig({
  shouldForwardProp: (prop) => prop !== 'primary'
})`
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.8);
  font-size: ${props => props.primary ? '1.2rem' : '0.9rem'};
  cursor: pointer;
  padding: ${props => props.primary ? '8px' : '6px'};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  min-width: ${props => props.primary ? '36px' : '28px'};
  min-height: ${props => props.primary ? '36px' : '28px'};
  
  &:hover {
    color: white;
    background: rgba(255, 255, 255, 0.1);
    transform: scale(1.05);
  }
  
  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
    transform: none;
  }
    ${props => props.primary && `
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 
      0 4px 15px rgba(102, 126, 234, 0.4),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
    min-width: 44px;
    min-height: 44px;
    
    &:hover {
      transform: scale(1.1);
      box-shadow: 
        0 6px 20px rgba(102, 126, 234, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
      background: linear-gradient(135deg, #7c8ef0 0%, #8b5fc7 100%);
    }
    
    &:active {
      transform: scale(0.98);
    }
  `}
`;

const VolumeSection = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 80px;
`;

const VolumeSlider = styled.input`
  width: 50px;
  height: 3px;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.2);
  outline: none;
  -webkit-appearance: none;
  cursor: pointer;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  }
  
  &::-moz-range-thumb {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    border: none;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
  }
`;

const KeyboardShortcuts = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
  padding: 8px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 6px;
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.5);
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const ShortcutKey = styled.span`
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 4px;
  border-radius: 3px;
  font-family: monospace;
  color: #667eea;
`;

const BattleAudioPlayer = ({ 
  track,
  onPlayStateChange,
  showKeyboardShortcuts = false
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  const audioRef = useRef(null);

  // Initialize audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);

  // Load track
  useEffect(() => {
    if (track?.url && audioRef.current) {
      setIsLoading(true);
      audioRef.current.src = track.url;
      audioRef.current.load();
    }
  }, [track?.url]);

  // Audio event handlers
  const handleLoadedMetadata = useCallback(() => {
    setDuration(audioRef.current?.duration || 0);
    setIsLoading(false);
  }, []);

  const handleTimeUpdate = useCallback(() => {
    setCurrentTime(audioRef.current?.currentTime || 0);
  }, []);

  const handleEnded = useCallback(() => {
    setIsPlaying(false);
    onPlayStateChange?.(false, track);
  }, [onPlayStateChange, track]);

  const handlePlay = useCallback(() => {
    setIsPlaying(true);
    onPlayStateChange?.(true, track);
  }, [onPlayStateChange, track]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
    onPlayStateChange?.(false, track);
  }, [onPlayStateChange, track]);

  // Attach audio event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
    };
  }, [handleLoadedMetadata, handleTimeUpdate, handleEnded, handlePlay, handlePause]);

  // Control functions
  const togglePlayPause = useCallback(() => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(console.error);
    }
  }, [isPlaying]);

  const rewind = useCallback((seconds = 10) => {
    if (!audioRef.current) return;
    
    const newTime = Math.max(0, currentTime - seconds);
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    
    toast.success(`⏪ -${seconds}s`, {
      duration: 800,
      style: { fontSize: '0.8rem' }
    });
  }, [currentTime]);

  const fastForward = useCallback((seconds = 10) => {
    if (!audioRef.current) return;
    
    const newTime = Math.min(duration, currentTime + seconds);
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    
    toast.success(`⏩ +${seconds}s`, {
      duration: 800,
      style: { fontSize: '0.8rem' }
    });
  }, [currentTime, duration]);

  const handleSeek = (e) => {
    if (!audioRef.current || duration === 0) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const newTime = percentage * duration;
    
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
    setIsMuted(newVolume === 0);
  };

  const toggleMute = useCallback(() => {
    if (audioRef.current) {
      const newMuted = !isMuted;
      setIsMuted(newMuted);
      audioRef.current.volume = newMuted ? 0 : volume;
    }
  }, [isMuted, volume]);

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (!track?.url) {
    return (
      <BattlePlayerContainer>
        <div style={{ 
          textAlign: 'center', 
          color: 'rgba(255, 255, 255, 0.5)',
          padding: '20px',
          fontSize: '0.9rem'
        }}>
          No audio available
        </div>
      </BattlePlayerContainer>
    );
  }

  return (
    <BattlePlayerContainer
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <audio ref={audioRef} preload="metadata" />
        <PlayerHeader>
        <TrackArtwork>
          {track.artwork || '🎵'}
        </TrackArtwork>
        <TrackInfo>
          <TrackTitle>{track.title}</TrackTitle>
          <TrackArtist>{track.artist}</TrackArtist>
        </TrackInfo>
      </PlayerHeader>

      <ProgressSection>
        <ProgressBar onClick={handleSeek}>
          <ProgressFill
            style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
            initial={{ width: 0 }}
            animate={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
            transition={{ duration: 0.1 }}
          />
        </ProgressBar>
        <TimeDisplay>
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </TimeDisplay>
      </ProgressSection>

      <ControlsSection>
        <MainControls>
          <ControlButton 
            onClick={() => rewind(10)}
            disabled={isLoading || currentTime < 10}
            title="Rewind 10s"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <FiSkipBack />
          </ControlButton>
          
          <ControlButton 
            primary
            onClick={togglePlayPause}
            disabled={isLoading}
            title={isPlaying ? "Pause" : "Play"}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isLoading ? '⋯' : isPlaying ? <FiPause /> : <FiPlay />}
          </ControlButton>

          <ControlButton 
            onClick={() => fastForward(10)}
            disabled={isLoading || currentTime > duration - 10}
            title="Forward 10s"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <FiSkipForward />
          </ControlButton>
        </MainControls>

        <VolumeSection>
          <ControlButton onClick={toggleMute} title="Toggle Mute">
            {isMuted || volume === 0 ? <FiVolumeX /> : <FiVolume2 />}
          </ControlButton>
          <VolumeSlider
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
          />
        </VolumeSection>
      </ControlsSection>

      {showKeyboardShortcuts && (
        <KeyboardShortcuts>
          <ShortcutKey>Space</ShortcutKey>: Play/Pause
          <ShortcutKey>J</ShortcutKey>: -10s
          <ShortcutKey>L</ShortcutKey>: +10s
          <ShortcutKey>M</ShortcutKey>: Mute
        </KeyboardShortcuts>
      )}
    </BattlePlayerContainer>
  );
};

export default BattleAudioPlayer;
