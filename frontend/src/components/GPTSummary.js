import React from 'react';
import { Typography, Paper } from '@mui/material';

function GPTSummary({ summary }) {
  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>GPT 요약</Typography>
      <Typography>{summary}</Typography>
    </Paper>
  );
}

export default GPTSummary;