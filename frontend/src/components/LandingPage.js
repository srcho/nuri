import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Container, Chip } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import Logo from '../DBpia_logo.png'; // 로고 파일을 import

function LandingPage() {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const handleSearch = () => {
    navigate('/search', { state: { query } });
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <img src={Logo} alt="DBpia Logo" style={{ height: '20px', marginRight: '16px' }} />
        </Box>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom>
            더 똑똑하게 찾는 학술 정보
          </Typography>
          <Box sx={{ my: 4 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="어떤 지식을 알고 싶으신가요?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              InputProps={{
                endAdornment: (
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSearch}
                    sx={{ ml: 1 }}
                  >
                    검색
                  </Button>
                ),
              }}
            />
          </Box>
          <Box sx={{ my: 2 }}>
            {['소셜미디어', '코로나19', 'AI', 'OTT', '인공지능', '유튜브'].map((tag) => (
              <Chip key={tag} label={tag} sx={{ m: 0.5 }} onClick={() => setQuery(tag)} />
            ))}
          </Box>
          <Typography variant="body2" sx={{ mt: 2 }}>
            프로토타입 테스트로, 신문방송학 관련 질문만 가능합니다.
          </Typography>
          <Typography variant="body2">
            주요 연구 키워드를 참고하여 질문해 보세요.
          </Typography>
        </Box>
      </Box>
    </Container>
  );
}

export default LandingPage;