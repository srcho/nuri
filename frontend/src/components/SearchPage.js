import React, { useState, useEffect, useCallback } from 'react';
import { Container, Typography, Box, Divider, Grid, CircularProgress, Button } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { useLocation, useNavigate } from 'react-router-dom';
import NoResult from './NoResult';
import GPTSummary from './GPTSummary';
import PaperList from './PaperList';
import { debounce } from 'lodash';

const SearchPage = () => {
  const [searchResult, setSearchResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const query = location.state?.query || "";

  console.log('SearchPage rendered');

  const debouncedHandleSearch = useCallback(
    (searchQuery) => {
      const search = async () => {
        setIsLoading(true);
        setError(null);

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
      };

      debounce(search, 300)();
    },
    []
  );

  useEffect(() => {
    console.log('SearchPage useEffect triggered', { query });
    if (query) {
      debouncedHandleSearch(query);
    }
  }, [query, debouncedHandleSearch]);

  const gptSaysNoData = searchResult?.answer?.toLowerCase().includes("nodata");

  // 유사도 60% 이상인 논문만 필터링 (이미 백엔드에서 처리됨)
  const filteredPapers = searchResult?.sources || [];

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
              {error && <Typography color="error" align="center">{error}</Typography>}
              {isLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                  <CircularProgress />
                </Box>
              ) : (
                <>
                  {gptSaysNoData && <NoResult />}
                  {!gptSaysNoData && searchResult && <GPTSummary summary={searchResult.answer} sources={filteredPapers} />}
                </>
              )}
            </Box>
          </Grid>
          <Grid item xs={4}>
            {filteredPapers?.length > 0 ? (
              <PaperList papers={filteredPapers} gptAnswer={searchResult?.answer} />
            ) : (
              !error && !isLoading && <Typography align="center">유사도가 60% 이상인 논문이 없습니다.</Typography>
            )}
          </Grid>
        </Grid>
      </Box>
    </Container>
  );

}

export default SearchPage;