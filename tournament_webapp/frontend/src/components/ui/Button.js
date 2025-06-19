import React from 'react';
import styled, { css } from 'styled-components';
import { motion } from 'framer-motion';

const ButtonBase = styled(motion.button).withConfig({
  shouldForwardProp: (prop) => !['variant', 'size', 'fullWidth'].includes(prop)
})`
  position: relative;
  font-family: var(--font-subtitle);
  font-size: 1.1rem;
  letter-spacing: 1px;
  padding: 10px 20px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-medium);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  text-transform: uppercase;
  border: none;
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  /* Ripple effect */
  &::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(255, 255, 255, 0.5);
    opacity: 0;
    border-radius: 100%;
    transform: scale(1, 1) translate(-50%);
    transform-origin: 50% 50%;
  }
  
  &:focus:not(:active)::after {
    animation: ripple 1s ease-out;
  }
  
  @keyframes ripple {
    0% {
      transform: scale(0, 0);
      opacity: 0.5;
    }
    20% {
      transform: scale(25, 25);
      opacity: 0.3;
    }
    100% {
      opacity: 0;
      transform: scale(40, 40);
    }
  }
  
  ${props => props.variant === 'primary' && css`
    background: linear-gradient(to bottom, var(--button-hover), var(--button-primary));
    color: white;
    text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.3);
    box-shadow: 
      0 4px 12px rgba(255, 69, 0, 0.25),
      inset 0 1px 1px rgba(255, 255, 255, 0.2);
    
    &:hover:not(:disabled) {
      box-shadow: 
        0 6px 16px rgba(255, 69, 0, 0.35),
        inset 0 1px 1px rgba(255, 255, 255, 0.4);
      transform: translateY(-2px);
    }
    
    &:active:not(:disabled) {
      box-shadow: 
        0 2px 8px rgba(255, 69, 0, 0.2),
        inset 0 1px 1px rgba(255, 255, 255, 0.1);
      transform: translateY(1px);
    }
  `}
  
  ${props => props.variant === 'secondary' && css`
    background: rgba(40, 40, 40, 0.6);
    backdrop-filter: blur(5px);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    
    &:hover:not(:disabled) {
      background: rgba(60, 60, 60, 0.7);
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
      transform: translateY(-2px);
    }
    
    &:active:not(:disabled) {
      background: rgba(30, 30, 30, 0.8);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      transform: translateY(1px);
    }
  `}
  
  ${props => props.variant === 'outline' && css`
    background: transparent;
    color: var(--button-primary);
    border: 2px solid var(--button-primary);
    
    &:hover:not(:disabled) {
      background: rgba(255, 69, 0, 0.1);
      transform: translateY(-2px);
    }
    
    &:active:not(:disabled) {
      background: rgba(255, 69, 0, 0.2);
      transform: translateY(1px);
    }
  `}
  
  ${props => props.size === 'small' && css`
    padding: 6px 14px;
    font-size: 0.9rem;
  `}
  
  ${props => props.size === 'large' && css`
    padding: 12px 24px;
    font-size: 1.3rem;
  `}
  
  ${props => props.fullWidth && css`
    width: 100%;
  `}
`;

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'medium',
  fullWidth = false,
  whileHover = { scale: 1.03 },
  whileTap = { scale: 0.97 },
  disabled = false,
  onClick,
  ...props 
}) => {
  return (
    <ButtonBase
      variant={variant}
      size={size}
      fullWidth={fullWidth}
      whileHover={whileHover}
      whileTap={whileTap}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </ButtonBase>
  );
};

export default Button;
