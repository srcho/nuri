import React from 'react';
import { Typography, Box } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

function NoResult() {
  return (
    <Box sx={{ textAlign: 'center', mt: 4 }}>
      <ErrorOutlineIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
      <Typography variant="h6" sx={{ mt: 2 }}>
        질문에 관련된 논문을 찾지 못했어요.
      </Typography>
      <Typography variant="body1" sx={{ mt: 1 }}>
        다른 질문으로 다시 시도해보세요!
      </Typography>
    </Box>
  );
}

export default NoResult;