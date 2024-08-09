import React, { useState } from 'react';
import { Typography, Paper, List, ListItem, ListItemText, Link, Box, Button } from '@mui/material';

const PaperList = ({ papers }) => {
  const [expandedAbstracts, setExpandedAbstracts] = useState({});

  const toggleAbstract = (index) => {
    setExpandedAbstracts(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>논문 리스트</Typography>
      <List>
        {papers.map((paper, index) => (
          <ListItem key={index} alignItems="flex-start" sx={{ px: 0 }}>
            <ListItemText
              primary={
                <Link href={paper.url} target="_blank" rel="noopener noreferrer" color="primary">
                  {`${index + 1}. ${paper.title}`}
                </Link>
              }
              secondary={
                <>
                  <Typography component="span" variant="body2" color="text.primary">
                    저자: {paper.authors === 'nan' ? '정보 없음' : paper.authors}
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        display: '-webkit-box',
                        WebkitLineClamp: expandedAbstracts[index] ? 'unset' : 4,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {paper.abstract}
                    </Typography>
                    <Button 
                      onClick={() => toggleAbstract(index)} 
                      sx={{ mt: 1, p: 0, minWidth: 'unset' }}
                    >
                      {expandedAbstracts[index] ? '접기' : '더보기'}
                    </Button>
                  </Box>
                  <Typography variant="body2" fontWeight="bold" color="text.secondary" sx={{ mt: 1 }}>
                    유사도: {paper.similarity != null ? `${(paper.similarity * 100).toFixed(2)}%` : '정보 없음'}
                  </Typography>
                </>
              }
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
}

export default PaperList;