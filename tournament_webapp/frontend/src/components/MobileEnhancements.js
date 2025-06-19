import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { FiMenu, FiX, FiVolumeX, FiVolume2, FiSmartphone, FiTablet } from 'react-icons/fi';

const MobileEnhancementsProvider = styled.div`
  /* Touch-friendly sizing */
  * {
    -webkit-touch-callout: none;
    -webkit-user-select: none;
    -khtml-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
  }
  
  /* Improve touch targets */
  button, a, input, select {
    min-height: 44px;
    min-width: 44px;
  }
  
  /* Smooth scrolling */
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  
  /* Prevent zoom on input focus */
  input, select, textarea {
    font-size: 16px;
  }
`;

const TouchIndicator = styled(motion.div)`
  position: fixed;
  pointer-events: none;
  width: 40px;
  height: 40px;
  border: 2px solid var(--primary-gold);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  z-index: 10000;
  opacity: 0;
`;

const MobileOptimizationOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(10px);
  z-index: 9999;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
`;

const MobileMessage = styled.div`
  text-align: center;
  color: white;
  max-width: 400px;
`;

const MobileTitle = styled.h2`
  color: var(--primary-gold);
  font-size: 1.8rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  justify-content: center;
`;

const MobileDescription = styled.p`
  font-size: 1.1rem;
  line-height: 1.6;
  margin-bottom: 30px;
  opacity: 0.9;
`;

const MobileActionButton = styled(motion.button)`
  background: linear-gradient(135deg, var(--primary-gold), var(--hiphop-orange));
  color: black;
  border: none;
  padding: 15px 30px;
  border-radius: 25px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  margin: 10px;
  min-width: 200px;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
  }
`;

const OrientationDetector = styled.div`
  @media screen and (orientation: landscape) and (max-height: 600px) {
    .landscape-warning {
      display: block;
    }
  }
  
  @media screen and (orientation: portrait) {
    .landscape-warning {
      display: none;
    }
  }
`;

const LandscapeWarning = styled(motion.div)`
  position: fixed;
  top: 20px;
  left: 20px;
  right: 20px;
  background: rgba(255, 193, 7, 0.95);
  color: black;
  padding: 15px;
  border-radius: 10px;
  z-index: 9998;
  text-align: center;
  font-weight: 600;
  display: none;
  
  &.landscape-warning {
    display: block;
  }
`;

const SwipeGestureDetector = styled.div`
  touch-action: pan-y;
  
  /* Disable pull-to-refresh */
  overscroll-behavior-y: contain;
`;

const VibrationFeedback = {
  light: () => {
    if (navigator.vibrate) {
      navigator.vibrate(50);
    }
  },
  medium: () => {
    if (navigator.vibrate) {
      navigator.vibrate(100);
    }
  },
  heavy: () => {
    if (navigator.vibrate) {
      navigator.vibrate([100, 50, 100]);
    }
  }
};

const MobileEnhancements = ({ children, showOptimizationOverlay = false }) => {
  const [touchPosition, setTouchPosition] = useState({ x: 0, y: 0, visible: false });
  const [isLandscape, setIsLandscape] = useState(false);
  const [showLandscapeWarning, setShowLandscapeWarning] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showMobileOverlay, setShowMobileOverlay] = useState(false);
  const touchTimeoutRef = useRef(null);

  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isSmallScreen = window.innerWidth <= 768;
      const hasTouchScreen = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
      
      setIsMobile(isMobileDevice || (isSmallScreen && hasTouchScreen));
      
      if (showOptimizationOverlay && (isMobileDevice || isSmallScreen)) {
        setShowMobileOverlay(true);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, [showOptimizationOverlay]);

  // Orientation detection
  useEffect(() => {
    const handleOrientationChange = () => {
      const landscape = window.innerHeight < window.innerWidth;
      setIsLandscape(landscape);
      
      if (landscape && window.innerHeight < 600 && isMobile) {
        setShowLandscapeWarning(true);
        setTimeout(() => setShowLandscapeWarning(false), 4000);
      }
    };

    handleOrientationChange();
    window.addEventListener('resize', handleOrientationChange);
    window.addEventListener('orientationchange', handleOrientationChange);
    
    return () => {
      window.removeEventListener('resize', handleOrientationChange);
      window.removeEventListener('orientationchange', handleOrientationChange);
    };
  }, [isMobile]);

  // Touch feedback
  const handleTouchStart = (e) => {
    if (!isMobile) return;
    
    const touch = e.touches[0];
    setTouchPosition({
      x: touch.clientX,
      y: touch.clientY,
      visible: true
    });

    // Light vibration feedback
    VibrationFeedback.light();

    // Clear existing timeout
    if (touchTimeoutRef.current) {
      clearTimeout(touchTimeoutRef.current);
    }

    // Hide touch indicator after 200ms
    touchTimeoutRef.current = setTimeout(() => {
      setTouchPosition(prev => ({ ...prev, visible: false }));
    }, 200);
  };

  const handleTouchEnd = () => {
    if (touchTimeoutRef.current) {
      clearTimeout(touchTimeoutRef.current);
    }
    setTouchPosition(prev => ({ ...prev, visible: false }));
  };

  // Prevent zoom on double tap
  useEffect(() => {
    let lastTouchEnd = 0;
    const handleTouchEndPreventZoom = (e) => {
      const now = (new Date()).getTime();
      if (now - lastTouchEnd <= 300) {
        e.preventDefault();
      }
      lastTouchEnd = now;
    };

    document.addEventListener('touchend', handleTouchEndPreventZoom, false);
    return () => document.removeEventListener('touchend', handleTouchEndPreventZoom);
  }, []);

  const optimizeBattles = () => {
    VibrationFeedback.medium();
    setShowMobileOverlay(false);
    
    // Enable additional mobile optimizations
    document.body.style.fontSize = '18px';
    document.body.style.lineHeight = '1.6';
    
    // Increase button sizes
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
      button.style.minHeight = '50px';
      button.style.padding = '12px 20px';
    });
  };

  const enableTabletMode = () => {
    VibrationFeedback.light();
    setShowMobileOverlay(false);
    
    // Tablet-specific optimizations
    document.body.style.fontSize = '16px';
    
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
      button.style.minHeight = '46px';
    });
  };

  return (
    <MobileEnhancementsProvider>
      <SwipeGestureDetector
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
      >
        <OrientationDetector>
          {children}
          
          {/* Touch indicator */}
          <AnimatePresence>
            {touchPosition.visible && (
              <TouchIndicator
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.7 }}
                exit={{ scale: 0, opacity: 0 }}
                style={{
                  left: touchPosition.x,
                  top: touchPosition.y
                }}
              />
            )}
          </AnimatePresence>

          {/* Landscape warning */}
          <LandscapeWarning
            className={showLandscapeWarning ? 'landscape-warning' : ''}
            initial={{ y: -100, opacity: 0 }}
            animate={showLandscapeWarning ? { y: 0, opacity: 1 } : { y: -100, opacity: 0 }}
            transition={{ type: 'spring', damping: 20 }}
          >
            📱 For the best experience, please rotate to portrait mode
          </LandscapeWarning>

          {/* Mobile optimization overlay */}
          <AnimatePresence>
            {showMobileOverlay && (
              <MobileOptimizationOverlay
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <MobileMessage>
                  <MobileTitle>
                    <FiSmartphone />
                    Mobile Tournament Experience
                  </MobileTitle>
                  <MobileDescription>
                    We've detected you're on a mobile device. Choose your preferred optimization mode for the best tournament experience.
                  </MobileDescription>
                  
                  <div>
                    <MobileActionButton
                      onClick={optimizeBattles}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <FiSmartphone style={{ marginRight: '8px' }} />
                      Optimize for Phone
                    </MobileActionButton>
                    
                    <MobileActionButton
                      onClick={enableTabletMode}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <FiTablet style={{ marginRight: '8px' }} />
                      Tablet Mode
                    </MobileActionButton>
                    
                    <MobileActionButton
                      onClick={() => setShowMobileOverlay(false)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      style={{ 
                        background: 'transparent',
                        border: '2px solid var(--primary-gold)',
                        color: 'var(--primary-gold)'
                      }}
                    >
                      Continue as Desktop
                    </MobileActionButton>
                  </div>
                </MobileMessage>
              </MobileOptimizationOverlay>
            )}
          </AnimatePresence>
        </OrientationDetector>
      </SwipeGestureDetector>
    </MobileEnhancementsProvider>
  );
};

// Hook for mobile features
export const useMobileFeatures = () => {
  const [isMobile, setIsMobile] = useState(false);
  const [orientation, setOrientation] = useState('portrait');
  
  useEffect(() => {
    const checkMobile = () => {
      const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isSmallScreen = window.innerWidth <= 768;
      setIsMobile(isMobileDevice || isSmallScreen);
      
      setOrientation(window.innerHeight > window.innerWidth ? 'portrait' : 'landscape');
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    window.addEventListener('orientationchange', checkMobile);
    
    return () => {
      window.removeEventListener('resize', checkMobile);
      window.removeEventListener('orientationchange', checkMobile);
    };
  }, []);

  const hapticFeedback = (type = 'light') => {
    VibrationFeedback[type]?.();
  };

  const isLandscape = orientation === 'landscape';
  const isPortrait = orientation === 'portrait';

  return {
    isMobile,
    isLandscape,
    isPortrait,
    orientation,
    hapticFeedback
  };
};

export { VibrationFeedback };
export default MobileEnhancements;
