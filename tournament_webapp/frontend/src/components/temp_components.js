/* This is for the FeatureCard component */
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

/* This is for the ChainDivider component */
const ChainDivider = styled.div`
  height: 1px;
  margin: 40px 0;
  background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.2), transparent);
  position: relative;
`;
