import React, { useState, useEffect, useCallback } from 'react';
import { Container, Typography, Box, TextField, IconButton, Divider, Grid, CircularProgress } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { useLocation } from 'react-router-dom';
import NoResult from './NoResult';
import GPTSummary from './GPTSummary';
import PaperList from './PaperList';

function SearchPage() {
  const [query, setQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const location = useLocation();

  const handleSearch = useCallback(async (searchQuery = query) => {
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
  }, [query]);

  useEffect(() => {
    if (location.state && location.state.query) {
      setQuery(location.state.query);
      handleSearch(location.state.query);
    }
  }, [location, handleSearch]);

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Grid container spacing={2}>
          <Grid item xs={8}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <SearchIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h5" component="h1" sx={{ fontWeight: 'bold' }}>
                {query || "연구 논문 검색"}
              </Typography>
            </Box>
            <Divider sx={{ borderColor: 'primary.main', borderWidth: 2 }} />
            <Box sx={{ mt: 2, mb: 4 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="질문을 입력하세요"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                InputProps={{
                  endAdornment: (
                    <IconButton onClick={() => handleSearch()} edge="end" color="primary">
                      <SearchIcon />
                    </IconButton>
                  ),
                }}
              />
            </Box>
            {isLoading && <CircularProgress />}
            {error && <Typography color="error" align="center">{error}</Typography>}
            {searchResult && <GPTSummary summary={searchResult.answer} />}
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