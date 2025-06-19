import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  FiPlay, 
  FiPause, 
  FiVolume2, 
  FiVolumeX, 
  FiRepeat, 
  FiShuffle,
  FiSkipBack,
  FiSkipForward,
  FiDownload,
  FiShare2
} from 'react-icons/fi';
import toast from 'react-hot-toast';

const PlayerContainer = styled(motion.div)`
  background: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  
  ${props => props.compact && `
    max-width: 400px;
  `}
  
  ${props => props.fullWidth && `
    width: 100%;
  `}
`;

const PlayerHeader = styled.div`
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  
  ${props => props.compact && `
    padding: 15px;
  `}
`;

const TrackInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const TrackArtwork = styled.div`
  width: 60px;
  height: 60px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--primary-gold), var(--hiphop-orange));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  flex-shrink: 0;
  
  ${props => props.compact && `
    width: 45px;
    height: 45px;
    font-size: 1.2rem;
  `}
`;

const TrackDetails = styled.div`
  flex: 1;
  min-width: 0;
`;

const TrackTitle = styled.h3`
  color: white;
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 4px 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  
  ${props => props.compact && `
    font-size: 1rem;
  `}
`;

const TrackArtist = styled.p`
  color: var(--text-light);
  font-size: 0.9rem;
  margin: 0;
  opacity: 0.8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const WaveformContainer = styled.div`
  padding: 20px;
  background: rgba(255, 255, 255, 0.02);
  
  ${props => props.compact && `
    padding: 15px;
  `}
`;

const WaveformCanvas = styled.canvas`
  width: 100%;
  height: 80px;
  cursor: pointer;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.05);
  
  ${props => props.compact && `
    height: 60px;
  `}
`;

const ProgressSection = styled.div`
  padding: 0 20px 10px;
  
  ${props => props.compact && `
    padding: 0 15px 10px;
  `}
`;

const ProgressBar = styled.div`
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  position: relative;
  cursor: pointer;
  margin-bottom: 8px;
  
  &:hover {
    background: rgba(255, 255, 255, 0.15);
  }
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, var(--primary-gold), var(--hiphop-orange));
  border-radius: 3px;
  position: relative;
  
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
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }
`;

const TimeDisplay = styled.div`
  display: flex;
  justify-content: space-between;
  color: var(--text-light);
  font-size: 0.8rem;
  font-family: monospace;
`;

const ControlsSection = styled.div`
  padding: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  
  ${props => props.compact && `
    padding: 15px;
  `}
`;

const MainControls = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
`;

const ControlButton = styled(motion.button)`
  background: none;
  border: none;
  color: var(--text-light);
  font-size: 1.2rem;
  cursor: pointer;
  padding: 8px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  
  &:hover {
    color: white;
    background: rgba(255, 255, 255, 0.1);
  }
  
  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
  
  ${props => props.primary && `
    background: linear-gradient(135deg, var(--primary-gold), var(--hiphop-orange));
    color: black;
    font-size: 1.4rem;
    width: 50px;
    height: 50px;
    
    &:hover {
      transform: scale(1.05);
      color: black;
    }
  `}
  
  ${props => props.active && `
    color: var(--primary-gold);
  `}
`;

const VolumeSection = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 120px;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const VolumeSlider = styled.input`
  width: 80px;
  height: 4px;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.2);
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--primary-gold);
    cursor: pointer;
  }
  
  &::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--primary-gold);
    cursor: pointer;
    border: none;
  }
`;

const SecondaryControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ComparisonMode = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 15px;
  }
`;

const ComparisonLabel = styled.div`
  text-align: center;
  color: var(--text-light);
  font-weight: 600;
  margin-bottom: 10px;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  font-size: 0.9rem;
`;

const KeyboardShortcuts = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  font-size: 0.75rem;
  color: var(--text-light);
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const ShortcutKey = styled.span`
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  color: var(--primary-gold);
`;

const AudioPlayer = ({ 
  tracks = [], 
  initialTrack = 0,
  compact = false,
  fullWidth = false,
  comparisonMode = false,
  onTrackChange,
  onPlayStateChange,
  showWaveform = true,
  showControls = true,
  autoPlay = false,
  loop = false,
  shuffle = false
}) => {
  const [currentTrackIndex, setCurrentTrackIndex] = useState(initialTrack);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLooping, setIsLooping] = useState(loop);
  const [isShuffling, setIsShuffling] = useState(shuffle);
  const [waveformData, setWaveformData] = useState([]);
  
  const audioRef = useRef(null);  const canvasRef = useRef(null);
  
  const currentTrack = useMemo(() => tracks[currentTrackIndex] || {}, [tracks, currentTrackIndex]);

  // Initialize audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
      audioRef.current.loop = isLooping;
    }
  }, [volume, isLooping]);

  // Load track
  useEffect(() => {
    if (currentTrack.url && audioRef.current) {
      setIsLoading(true);
      audioRef.current.src = currentTrack.url;
      audioRef.current.load();
      
      if (autoPlay) {
        audioRef.current.play().catch(console.error);
      }
    }
  }, [currentTrack.url, autoPlay]);  // Audio event handlers
  const handleLoadedMetadata = useCallback(() => {
    setDuration(audioRef.current?.duration || 0);
    setIsLoading(false);
    // Generate waveform when metadata loads
    const points = compact ? 50 : 100;
    const data = Array.from({ length: points }, () => Math.random() * 0.8 + 0.1);
    setWaveformData(data);
  }, [compact]);

  const handleTimeUpdate = useCallback(() => {
    setCurrentTime(audioRef.current?.currentTime || 0);
  }, []);
  const handleEnded = useCallback(() => {
    if (!isLooping) {
      // Move to next track
      const newIndex = isShuffling 
        ? Math.floor(Math.random() * tracks.length)
        : (currentTrackIndex + 1) % tracks.length;
      
      setCurrentTrackIndex(newIndex);
      onTrackChange?.(tracks[newIndex], newIndex);
    }
  }, [isLooping, isShuffling, tracks, currentTrackIndex, onTrackChange]);

  const handlePlay = useCallback(() => {
    setIsPlaying(true);
    onPlayStateChange?.(true, currentTrack);
  }, [onPlayStateChange, currentTrack]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
    onPlayStateChange?.(false, currentTrack);
  }, [onPlayStateChange, currentTrack]);

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
    };  }, [handleLoadedMetadata, handleTimeUpdate, handleEnded, handlePlay, handlePause]);

  // Draw waveform
  useEffect(() => {
    if (!showWaveform || !canvasRef.current || waveformData.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();
    
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    ctx.clearRect(0, 0, width, height);

    const barWidth = width / waveformData.length;
    const progress = duration > 0 ? currentTime / duration : 0;

    waveformData.forEach((amplitude, index) => {
      const barHeight = amplitude * height * 0.8;
      const x = index * barWidth;
      const y = (height - barHeight) / 2;
      
      // Color based on progress
      const isPlayed = index / waveformData.length < progress;
      ctx.fillStyle = isPlayed 
        ? 'rgba(255, 215, 0, 0.8)' 
        : 'rgba(255, 255, 255, 0.3)';
      
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    });
  }, [waveformData, currentTime, duration, showWaveform]);  // Control functions
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
    
    // Visual feedback
    toast.success(`⏪ Rewound ${seconds}s`, {
      duration: 1000,
      style: { fontSize: '0.9rem' }
    });
  }, [currentTime]);

  const fastForward = useCallback((seconds = 10) => {
    if (!audioRef.current) return;
    
    const newTime = Math.min(duration, currentTime + seconds);
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    
    // Visual feedback
    toast.success(`⏩ Forward ${seconds}s`, {
      duration: 1000,
      style: { fontSize: '0.9rem' }
    });
  }, [currentTime, duration]);

  const handlePrevious = () => {
    const newIndex = isShuffling 
      ? Math.floor(Math.random() * tracks.length)
      : (currentTrackIndex - 1 + tracks.length) % tracks.length;
    
    setCurrentTrackIndex(newIndex);
    onTrackChange?.(tracks[newIndex], newIndex);
  };

  const handleNext = () => {
    const newIndex = isShuffling 
      ? Math.floor(Math.random() * tracks.length)
      : (currentTrackIndex + 1) % tracks.length;
    
    setCurrentTrackIndex(newIndex);
    onTrackChange?.(tracks[newIndex], newIndex);
  };

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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Only handle shortcuts when audio player is focused or no other input is focused
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') {
        return;
      }

      switch (e.key) {
        case ' ':
        case 'k':
          e.preventDefault();
          togglePlayPause();
          break;
        case 'j':
          e.preventDefault();
          rewind(10);
          break;
        case 'l':
          e.preventDefault();
          fastForward(10);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          rewind(5);
          break;
        case 'ArrowRight':
          e.preventDefault();
          fastForward(5);
          break;
        case 'm':
          e.preventDefault();
          toggleMute();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [togglePlayPause, rewind, fastForward, toggleMute]);

  const handleDownload = () => {
    if (currentTrack.downloadUrl) {
      window.open(currentTrack.downloadUrl, '_blank');
    } else {
      toast.success('Download feature coming soon!');
    }
  };

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: currentTrack.title,
          text: `Check out this track: ${currentTrack.title}`,
          url: window.location.href
        });
      } else {
        await navigator.clipboard.writeText(window.location.href);
        toast.success('Link copied to clipboard!');
      }
    } catch (error) {
      toast.error('Failed to share');
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (comparisonMode && tracks.length >= 2) {
    return (
      <PlayerContainer compact={compact} fullWidth={fullWidth}>
        <ComparisonMode>
          {tracks.slice(0, 2).map((track, index) => (
            <div key={index}>
              <ComparisonLabel>
                {track.label || `Model ${index + 1}`}
              </ComparisonLabel>
              <AudioPlayer
                tracks={[track]}
                compact={true}
                showWaveform={showWaveform}
                onPlayStateChange={(playing, trackData) => {
                  // Pause other player when this one starts
                  if (playing && onPlayStateChange) {
                    onPlayStateChange(playing, trackData, index);
                  }
                }}
              />
            </div>
          ))}
        </ComparisonMode>
      </PlayerContainer>
    );
  }

  return (
    <PlayerContainer 
      compact={compact} 
      fullWidth={fullWidth}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <audio ref={audioRef} preload="metadata" />
      
      {showControls && (
        <PlayerHeader compact={compact}>
          <TrackInfo>
            <TrackArtwork compact={compact}>
              🎵
            </TrackArtwork>
            <TrackDetails>
              <TrackTitle compact={compact}>
                {currentTrack.title || 'Unknown Track'}
              </TrackTitle>
              <TrackArtist>
                {currentTrack.artist || 'AI Generated Mix'}
              </TrackArtist>
            </TrackDetails>
          </TrackInfo>
        </PlayerHeader>
      )}

      {showWaveform && (
        <WaveformContainer compact={compact}>
          <WaveformCanvas 
            ref={canvasRef}
            compact={compact}
            onClick={handleSeek}
          />
        </WaveformContainer>
      )}

      <ProgressSection compact={compact}>
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

      {showControls && (
        <ControlsSection compact={compact}>          <MainControls>
            <ControlButton 
              onClick={() => setIsShuffling(!isShuffling)}
              active={isShuffling}
              disabled={tracks.length <= 1}
              title="Shuffle"
            >
              <FiShuffle />
            </ControlButton>
            
            <ControlButton 
              onClick={handlePrevious}
              disabled={tracks.length <= 1}
              title="Previous Track"
            >
              <FiSkipBack />
            </ControlButton>

            <ControlButton 
              onClick={() => rewind(10)}
              disabled={isLoading || currentTime < 10}
              title="Rewind 10s"
            >
              ⏪
            </ControlButton>
            
            <ControlButton 
              primary
              onClick={togglePlayPause}
              disabled={isLoading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title={isPlaying ? "Pause" : "Play"}
            >
              {isLoading ? '...' : isPlaying ? <FiPause /> : <FiPlay />}
            </ControlButton>

            <ControlButton 
              onClick={() => fastForward(10)}
              disabled={isLoading || currentTime > duration - 10}
              title="Forward 10s"
            >
              ⏩
            </ControlButton>
            
            <ControlButton 
              onClick={handleNext}
              disabled={tracks.length <= 1}
            >
              <FiSkipForward />
            </ControlButton>
            
            <ControlButton 
              onClick={() => setIsLooping(!isLooping)}
              active={isLooping}
            >
              <FiRepeat />
            </ControlButton>
          </MainControls>

          <VolumeSection>
            <ControlButton onClick={toggleMute}>
              {isMuted || volume === 0 ? <FiVolumeX /> : <FiVolume2 />}
            </ControlButton>
            <VolumeSlider
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
            />
          </VolumeSection>

          <SecondaryControls>
            <ControlButton onClick={handleDownload}>
              <FiDownload />
            </ControlButton>
            <ControlButton onClick={handleShare}>
              <FiShare2 />
            </ControlButton>
          </SecondaryControls>
        </ControlsSection>
      )}

      {showControls && (
        <KeyboardShortcuts>
          <ShortcutKey>Space</ShortcutKey> / <ShortcutKey>K</ShortcutKey>: Play/Pause
          <ShortcutKey>J</ShortcutKey>: Rewind 10s
          <ShortcutKey>L</ShortcutKey>: Forward 10s
          <ShortcutKey>←</ShortcutKey>: Rewind 5s
          <ShortcutKey>→</ShortcutKey>: Forward 5s
          <ShortcutKey>M</ShortcutKey>: Toggle Mute
        </KeyboardShortcuts>
      )}
    </PlayerContainer>
  );
};

export default AudioPlayer;
