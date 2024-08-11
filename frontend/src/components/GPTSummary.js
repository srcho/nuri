import React, { useState, useEffect } from 'react';
import { Typography, Paper, Box, Tooltip } from '@mui/material';

function GPTSummary({ summary, sources = [] }) {
  const [processedSummary, setProcessedSummary] = useState('');
  const [footnotes, setFootnotes] = useState([]);

  useEffect(() => {
    if (summary) {
      const processedText = summary.replace(/\[(\d+)\]/g, (match, p1) => {
        const index = parseInt(p1, 10) - 1;
        if (index >= 0 && index < sources.length) {
          return `[${p1}]`;
        }
        return match;
      });

      setProcessedSummary(processedText);
    }

    if (Array.isArray(sources)) {
      setFootnotes(sources.map((source, index) => ({
        id: index + 1,
        title: source.title || "제목 없음",
        authors: source.authors || "저자 정보 없음"
      })));
    }
  }, [summary, sources]);

  if (!summary) {
    return (
      <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
        <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>DBpia AI</Typography>
        <Typography>답변을 불러오는 중 오류가 발생했습니다.</Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>DBpia AI</Typography>
      <Typography>
        {processedSummary.split(/(\[\d+\])/).map((part, index) => {
          if (part.match(/\[(\d+)\]/)) {
            const footnoteId = parseInt(part.match(/\[(\d+)\]/)[1], 10);
            const footnote = footnotes.find(f => f.id === footnoteId);
            return footnote ? (
              <Tooltip key={index} title={`${footnote.title} - ${footnote.authors}`} arrow>
                <sup style={{ cursor: 'pointer', color: 'blue' }}>{part}</sup>
              </Tooltip>
            ) : part;
          }
          return part;
        })}
      </Typography>
      {footnotes.length > 0 && (
        <Box mt={2}>
          <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>참고 문헌</Typography>
          {footnotes.map((footnote) => (
            <Typography key={footnote.id} variant="body2" paragraph>
              [{footnote.id}] {footnote.title} - {footnote.authors}
            </Typography>
          ))}
        </Box>
      )}
    </Paper>
  );
}

export default GPTSummary;