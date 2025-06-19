import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const InputContainer = styled.div`
  position: relative;
  margin-bottom: 20px;
  width: 100%;
`;

const Label = styled.label`
  font-size: 1rem;
  color: ${({ theme }) => theme.textMuted};
  margin-bottom: 6px;
  display: block;
  font-weight: 500;
`;

const InputWrapper = styled.div`
  position: relative;
  width: 100%;
`;

const StyledInput = styled.input`
  width: 100%;
  padding: 12px 16px;
  background: ${({ theme }) => theme.mode === 'dark' 
    ? 'rgba(30, 30, 30, 0.8)' 
    : 'rgba(255, 255, 255, 0.8)'};
  color: ${({ theme }) => theme.text};
  border: 1px solid ${({ theme }) => theme.border};
  border-radius: var(--radius-md);
  font-size: 1rem;
  font-family: inherit;
  transition: all var(--transition-medium);
  backdrop-filter: blur(5px);
  box-shadow: inset 0 2px 4px ${({ theme }) => theme.mode === 'dark' 
    ? 'rgba(0, 0, 0, 0.2)' 
    : 'rgba(0, 0, 0, 0.05)'};
  outline: none;
  
  &:focus {
    border-color: var(--button-primary);
    box-shadow: 0 0 0 2px rgba(255, 69, 0, 0.2);
  }
  
  &::placeholder {
    color: ${({ theme }) => theme.mode === 'dark' 
      ? 'rgba(255, 255, 255, 0.3)' 
      : 'rgba(0, 0, 0, 0.3)'};
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ErrorText = styled(motion.p)`
  color: var(--primary-red);
  font-size: 0.85rem;
  margin: 5px 0 0;
`;

const Icon = styled.span`
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: ${({ theme }) => theme.textMuted};
`;

const InputField = ({
  label,
  name,
  type = 'text',
  placeholder,
  value,
  onChange,
  error,
  icon,
  ...props
}) => {
  return (
    <InputContainer>
      {label && <Label htmlFor={name}>{label}</Label>}
      <InputWrapper>
        <StyledInput
          id={name}
          name={name}
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          {...props}
        />
        {icon && <Icon>{icon}</Icon>}
      </InputWrapper>
      {error && (
        <ErrorText
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {error}
        </ErrorText>
      )}
    </InputContainer>
  );
};

export default InputField;
