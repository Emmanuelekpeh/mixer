import React from 'react';
import styled, { css } from 'styled-components';
import { motion } from 'framer-motion';

const CardContainer = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => !['variant', 'clickable', 'canVote', 'isSelected'].includes(prop)
})`
  background: var(--glass-bg);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: var(--radius-lg);
  border: 1px solid var(--glass-border);
  box-shadow: var(--glass-shadow);
  padding: var(--space-lg);
  width: 100%;
  transition: all var(--transition-medium);
  overflow: hidden;
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(
      90deg,
      rgba(255, 255, 255, 0) 0%,
      rgba(255, 255, 255, 0.3) 50%,
      rgba(255, 255, 255, 0) 100%
    );
  }
  
  ${props => props.variant === 'elevated' && css`
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    
    &:hover {
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.2);
      transform: translateY(-3px);
    }
  `}
  
  ${props => props.variant === 'bordered' && css`
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.03);
  `}
  
  ${props => props.variant === 'colored' && css`
    background: linear-gradient(135deg, rgba(30, 144, 255, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
    border: 1px solid rgba(138, 43, 226, 0.2);
  `}
  
  ${props => props.clickable && css`
    cursor: pointer;
    
    &:hover {
      transform: translateY(-3px);
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.2);
    }
    
    &:active {
      transform: translateY(-1px);
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.15);
    }
  `}
`;

const CardHeader = styled.div`
  margin-bottom: var(--space-md);
  padding-bottom: var(--space-md);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const CardTitle = styled.h3`
  margin: 0;
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--text-light);
`;

const CardSubtitle = styled.p`
  margin: 0;
  font-size: 1rem;
  color: var(--text-muted);
  margin-top: var(--space-xs);
`;

const CardContent = styled.div`
  position: relative;
`;

const CardFooter = styled.div`
  margin-top: var(--space-lg);
  padding-top: var(--space-md);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: var(--space-md);
`;

const Card = ({ 
  children, 
  variant = 'default',
  title,
  subtitle,
  footer,
  clickable = false,
  onClick,
  ...props 
}) => {
  return (
    <CardContainer 
      variant={variant} 
      clickable={clickable}
      onClick={clickable ? onClick : undefined}
      whileHover={clickable ? { y: -3 } : {}}
      whileTap={clickable ? { y: -1 } : {}}
      {...props}
    >
      {(title || subtitle) && (
        <CardHeader>
          <div>
            {title && <CardTitle>{title}</CardTitle>}
            {subtitle && <CardSubtitle>{subtitle}</CardSubtitle>}
          </div>
        </CardHeader>
      )}
      
      <CardContent>{children}</CardContent>
      
      {footer && <CardFooter>{footer}</CardFooter>}
    </CardContainer>
  );
};

export default Card;
