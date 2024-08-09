import React, { useState } from 'react';
import { Container, Typography, Box, TextField, IconButton, Divider, Grid, CircularProgress } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import NoResult from './components/NoResult';
import GPTSummary from './components/GPTSummary';
import PaperList from './components/PaperList';

const theme = createTheme({
  palette: {
    primary: {
      main: '#E1261C',
    },
  },
});

function App() {
  const [query, setQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gptSummary, setGptSummary] = useState('');

  const handleSearch = async () => {
    setIsLoading(true);
    setError(null);
    setSearchResult(null);
    setGptSummary('');

    try {
      const searchResponse = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      });

      if (!searchResponse.ok) throw new Error('Search request failed');
      const searchData = await searchResponse.json();
      setSearchResult(searchData.sources);
      setGptSummary(searchData.answer);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
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
                      <IconButton onClick={handleSearch} edge="end" color="primary">
                        <SearchIcon />
                      </IconButton>
                    ),
                  }}
                />
              </Box>
              {isLoading && <CircularProgress />}
              {error && <Typography color="error" align="center">{error}</Typography>}
              {gptSummary && <GPTSummary summary={gptSummary} />}
            </Grid>
            <Grid item xs={4}>
              {searchResult?.length > 0 ? (
                <PaperList papers={searchResult} />
              ) : (
                !isLoading && !error && <NoResult />
              )}
            </Grid>
          </Grid>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;