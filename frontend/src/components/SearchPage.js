import React, { useState, useEffect, useCallback } from 'react';
import { Container, Typography, Box, Divider, Grid, CircularProgress, Button } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { useLocation, useNavigate } from 'react-router-dom';
import NoResult from './NoResult';
import GPTSummary from './GPTSummary';
import PaperList from './PaperList';

const SearchPage = () => {
  const [searchResult, setSearchResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const query = location.state?.query || "";

  const handleSearch = useCallback(async (searchQuery) => {
    setIsLoading(true);
    setError(null);
    setSearchResult(null);

    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: searchQuery }),
      });

      if (!response.ok) throw new Error('검색 요청 실패');
      const data = await response.json();
      setSearchResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (query) {
      handleSearch(query);
    }
  }, [query, handleSearch]);

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          메인 페이지로 돌아가기
        </Button>
        <Grid container spacing={2}>
          <Grid item xs={8}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <SearchIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h5" component="h1" sx={{ fontWeight: 'bold' }}>
                검색 결과: "{query}"
              </Typography>
            </Box>
            <Divider sx={{ borderColor: 'primary.main', borderWidth: 2 }} />
            <Box sx={{ mt: 4 }}>
              {isLoading && <CircularProgress />}
              {error && <Typography color="error" align="center">{error}</Typography>}
              {searchResult && <GPTSummary summary={searchResult.answer} />}
            </Box>
          </Grid>
          <Grid item xs={4}>
            {searchResult?.sources?.length > 0 ? (
              <PaperList papers={searchResult.sources} />
            ) : (
              !isLoading && !error && <NoResult />
            )}
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default SearchPage;