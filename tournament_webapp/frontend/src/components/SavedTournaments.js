import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { FiPlay, FiEye, FiTrash2, FiMusic, FiClock, FiAward, FiDownload } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { Card, Button } from './ui';

const SavedTournamentsContainer = styled(motion.div)`
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 40px;
`;

const Title = styled.h1`
  color: white;
  font-size: 2.5rem;
  font-weight: 800;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
  font-family: var(--font-title);
`;

const Subtitle = styled.p`
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.2rem;
  margin-bottom: 30px;
`;

const TournamentGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 25px;
  margin-bottom: 40px;
`;

const TournamentCard = styled(Card)`
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  }
`;

const TournamentHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 15px;
`;

const TournamentInfo = styled.div`
  flex: 1;
`;

const TournamentTitle = styled.h3`
  color: white;
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: 5px;
`;

const TournamentMeta = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.9rem;
  margin-bottom: 10px;
`;

const StatusBadge = styled.div`
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  
  ${props => props.status === 'completed' && `
    background: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
    border: 1px solid rgba(76, 175, 80, 0.3);
  `}
  
  ${props => props.status === 'active' && `
    background: rgba(255, 193, 7, 0.2);
    color: #FFC107;
    border: 1px solid rgba(255, 193, 7, 0.3);
  `}
  
  ${props => props.status === 'paused' && `
    background: rgba(255, 87, 34, 0.2);
    color: #FF5722;
    border: 1px solid rgba(255, 87, 34, 0.3);
  `}
`;

const TournamentStats = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 20px;
`;

const Stat = styled.div`
  text-align: center;
  
  .icon {
    color: #667eea;
    margin-bottom: 5px;
  }
  
  .value {
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 2px;
  }
  
  .label {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 10px;
`;

const ActionButton = styled(Button)`
  flex: 1;
  padding: 8px 16px;
  font-size: 0.9rem;
  
  &.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  &.secondary {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  &.danger {
    background: rgba(244, 67, 54, 0.2);
    border: 1px solid rgba(244, 67, 54, 0.3);
    color: #f44336;
    
    &:hover {
      background: rgba(244, 67, 54, 0.3);
    }
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 60px 20px;
  color: rgba(255, 255, 255, 0.6);
`;

const EmptyIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
  opacity: 0.5;
`;

const EmptyText = styled.h3`
  font-size: 1.5rem;
  margin-bottom: 10px;
  color: rgba(255, 255, 255, 0.8);
`;

const EmptySubtext = styled.p`
  font-size: 1rem;
  margin-bottom: 30px;
`;

const SavedTournaments = ({ user, onResumeTournament, onCreateNew }) => {  const [tournaments, setTournaments] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const loadSavedTournaments = useCallback(async () => {
    if (!user?.user_id) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      
      // First try to get tournaments from the API
      try {
        const [activeResponse, completedResponse] = await Promise.all([
          fetch(`/api/users/${user.user_id}/tournaments?status=active`),
          fetch(`/api/users/${user.user_id}/tournaments?status=completed`)
        ]);

        let apiTournaments = [];

        if (activeResponse.ok) {
          const activeData = await activeResponse.json();
          if (activeData.success) {
            apiTournaments = [...apiTournaments, ...activeData.tournaments];
          }
        }

        if (completedResponse.ok) {
          const completedData = await completedResponse.json();
          if (completedData.success) {
            apiTournaments = [...apiTournaments, ...completedData.tournaments];
          }
        }

        if (apiTournaments.length > 0) {
          // Sort by creation date (newest first)
          apiTournaments.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          setTournaments(apiTournaments);
          console.log(`📋 Loaded ${apiTournaments.length} tournaments from API`);
          return;
        }
      } catch (apiError) {
        console.warn('API fetch failed, falling back to localStorage:', apiError);
      }
      
      // Fallback: Get tournaments from localStorage
      const savedTournaments = [];
      
      // Check for tournament history in localStorage
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('tournament_')) {
          try {
            const tournamentData = JSON.parse(localStorage.getItem(key));
            if (tournamentData && tournamentData.user_id === user?.user_id) {
              savedTournaments.push({
                tournament_id: key,
                ...tournamentData,
                created_at: tournamentData.created_at || new Date().toISOString()
              });
            }
          } catch (err) {
            console.error('Error parsing tournament data:', err);
          }
        }
      }
      
      // Sort by creation date (newest first)
      savedTournaments.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      
      setTournaments(savedTournaments);
      console.log(`📋 Loaded ${savedTournaments.length} tournaments from localStorage`);
      
    } catch (error) {
      console.error('Error loading saved tournaments:', error);
      toast.error('Failed to load saved tournaments');
    } finally {
      setLoading(false);
    }
  }, [user?.user_id]);

  useEffect(() => {
    loadSavedTournaments();
  }, [loadSavedTournaments]);
  const handleResumeTournament = async (tournament) => {
    if (tournament.status === 'completed') {
      toast.error('Cannot resume completed tournament');
      return;
    }
    
    try {
      console.log('🔄 Attempting to resume tournament:', tournament.tournament_id);
      
      // First try to resume using the resume endpoint
      const resumeResponse = await fetch(`/api/tournaments/${tournament.tournament_id}/resume`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (resumeResponse.ok) {
        const resumeData = await resumeResponse.json();
        if (resumeData.success && resumeData.tournament) {
          console.log('✅ Tournament resumed successfully:', resumeData.tournament);
          toast.success('Tournament resumed! Continue where you left off.');
          
          // Update local storage and trigger resume
          localStorage.setItem('activeTournament', tournament.tournament_id);
          
          if (onResumeTournament) {
            onResumeTournament(resumeData.tournament);
          }
          
          navigate('/battle');
          return;
        }
      }
      
      // Fallback: try regular fetch
      const response = await fetch(`/api/tournaments/${tournament.tournament_id}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.tournament) {
          console.log('✅ Tournament loaded via regular fetch:', data.tournament);
          toast.success('Tournament loaded successfully');
          
          localStorage.setItem('activeTournament', tournament.tournament_id);
          
          if (onResumeTournament) {
            onResumeTournament(data.tournament);
          }
          
          navigate('/battle');
          return;
        }
      }
      
      // If all fails, use localStorage data
      if (onResumeTournament) {
        onResumeTournament(tournament);
        localStorage.setItem('activeTournament', tournament.tournament_id);
        toast.success(`Resumed tournament: ${tournament.name || 'Unnamed Tournament'}`);
        navigate('/battle');
      }
      
    } catch (error) {
      console.error('Error resuming tournament:', error);
      toast.error(`Failed to resume tournament: ${error.message}`);
      
      // Reload tournaments to refresh the list
      loadSavedTournaments();
    }
  };  const handleViewResults = (tournament) => {
    // Navigate to results page using tournament_id
    const tournamentId = tournament.tournament_id || tournament.id;
    navigate(`/results/${tournamentId}`);
  };  const handleDeleteTournament = async (tournament) => {
    if (window.confirm('Are you sure you want to delete this tournament? This action cannot be undone.')) {
      try {
        const tournamentId = tournament.tournament_id || tournament.id;
        const userId = user?.user_id || 'default_user';
        
        // Try to delete from API first
        try {
          const response = await fetch(`/api/tournaments/${tournamentId}`, {
            method: 'DELETE',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_id: userId })
          });

          if (response.ok) {
            const result = await response.json();
            console.log('✅ Tournament deleted from API:', result);
            toast.success('Tournament deleted successfully');
            
            // Refresh the tournaments list from API
            await loadSavedTournaments();
            return;
          } else {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            console.warn('❌ API delete failed:', response.status, errorData);
            toast.warning(`Tournament delete failed: ${errorData.detail || 'Server error'}`);
            
            // If API delete fails with 404 (not found), we should clean up local storage anyway
            if (response.status === 404) {
              console.log('Tournament not found in API, cleaning up local storage');
              localStorage.removeItem(tournamentId);
              localStorage.removeItem(`tournament_${tournamentId}`);
              setTournaments(prev => prev.filter(t => (t.tournament_id || t.id) !== tournamentId));
              return;
            }
          }
        } catch (apiError) {
          console.warn('❌ API delete error:', apiError);
          toast.warning('Tournament delete failed (API unavailable)');
        }
          // Always try to remove from localStorage as fallback
        localStorage.removeItem(tournament.id);
        localStorage.removeItem(`tournament_${tournamentId}`);
        
        // Update state to remove from UI
        setTournaments(prev => prev.filter(t => 
          t.id !== tournament.id && 
          t.tournament_id !== tournamentId
        ));
        
        // Try to reload tournaments from API after a short delay
        setTimeout(() => {
          loadSavedTournaments();
        }, 500);
        
      } catch (error) {
        console.error('Error deleting tournament:', error);
        toast.error('Failed to delete tournament');
      }
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <SavedTournamentsContainer>
        <div style={{ textAlign: 'center', padding: '60px 20px', color: 'white' }}>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            style={{ display: 'inline-block', marginBottom: '20px' }}
          >
            <FiMusic size={48} />
          </motion.div>
          <div>Loading your tournaments...</div>
        </div>
      </SavedTournamentsContainer>
    );
  }

  return (
    <SavedTournamentsContainer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <Header>
        <Title>Your Tournaments</Title>
        <Subtitle>Resume active tournaments or review your mixing history</Subtitle>
      </Header>

      {tournaments.length === 0 ? (
        <EmptyState>
          <EmptyIcon>
            <FiMusic />
          </EmptyIcon>
          <EmptyText>No tournaments yet</EmptyText>
          <EmptySubtext>Create your first AI mixing tournament to get started</EmptySubtext>
          <Button onClick={onCreateNew} className="primary">
            Create New Tournament
          </Button>
        </EmptyState>
      ) : (
        <>
          <TournamentGrid>
            <AnimatePresence>
              {tournaments.map((tournament, index) => (
                <motion.div
                  key={tournament.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                >
                  <TournamentCard>
                    <TournamentHeader>
                      <TournamentInfo>                        <TournamentTitle>
                          {tournament.original_filename || tournament.name || `Tournament ${(tournament.tournament_id || tournament.id)?.split('_')[1]?.slice(0, 8) || 'Unknown'}`}
                        </TournamentTitle>
                        <TournamentMeta>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                            <FiClock size={14} />
                            {formatDate(tournament.created_at)}
                          </div>
                          {tournament.victor_model_id && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                              <FiAward size={14} />
                              Champion: {tournament.victor_model_id}
                            </div>
                          )}
                        </TournamentMeta>
                      </TournamentInfo>
                      <StatusBadge status={tournament.status}>
                        {tournament.status}
                      </StatusBadge>
                    </TournamentHeader>                    <TournamentStats>
                      <Stat>
                        <div className="icon">
                          <FiMusic size={18} />
                        </div>
                        <div className="value">{tournament.current_round || 1}</div>
                        <div className="label">Round</div>
                      </Stat>
                      <Stat>
                        <div className="icon">
                          <FiAward size={18} />
                        </div>
                        <div className="value">{tournament.pairs_completed || 0}</div>
                        <div className="label">Completed</div>
                      </Stat>
                      <Stat>
                        <div className="icon">
                          <FiDownload size={18} />
                        </div>
                        <div className="value">{tournament.max_rounds || 5}</div>
                        <div className="label">Max Rounds</div>
                      </Stat>
                    </TournamentStats>

                    <ActionButtons>
                      {tournament.status === 'active' ? (
                        <ActionButton 
                          className="primary"
                          onClick={() => handleResumeTournament(tournament)}
                        >
                          <FiPlay size={14} style={{ marginRight: '5px' }} />
                          Resume
                        </ActionButton>
                      ) : (
                        <ActionButton 
                          className="secondary"
                          onClick={() => handleViewResults(tournament)}
                        >
                          <FiEye size={14} style={{ marginRight: '5px' }} />
                          View Results
                        </ActionButton>
                      )}
                      
                      <ActionButton 
                        className="danger"
                        onClick={() => handleDeleteTournament(tournament)}
                      >
                        <FiTrash2 size={14} />
                      </ActionButton>
                    </ActionButtons>
                  </TournamentCard>
                </motion.div>
              ))}
            </AnimatePresence>
          </TournamentGrid>

          <div style={{ textAlign: 'center' }}>
            <Button onClick={onCreateNew} className="primary">
              Create New Tournament
            </Button>
          </div>
        </>
      )}
    </SavedTournamentsContainer>
  );
};

export default SavedTournaments;
