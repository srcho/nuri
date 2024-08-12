import React, { useState } from 'react';
import { Typography, Paper, List, ListItem, ListItemText, Link, Box, Button } from '@mui/material';

const PaperList = ({ papers, gptAnswer }) => {
  const [expandedAbstracts, setExpandedAbstracts] = useState({});

  const toggleAbstract = (index) => {
    setExpandedAbstracts(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  // 필터링은 이미 백엔드에서 수행되었으므로 여기서는 필요 없음
  const filteredPapers = papers;

  // 필터링된 논문 리스트가 비어있는 경우
  if (filteredPapers.length === 0) {
    return null; // 아무것도 표시되지 않음
  }

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ color: 'primary.main' }}>검색된 논문</Typography>
      </Box>
      <List>
        {filteredPapers.map((paper, index) => (
          <ListItem key={index} alignItems="flex-start" sx={{ px: 0 }}>
            <ListItemText
              primary={
                <Typography variant="subtitle1" color="primary">
                  [{index + 1}] {
                    paper.url && paper.url !== "정보 없음" ? (
                      <Link href={paper.url} target="_blank" rel="noopener noreferrer">
                        {paper.title || "제목 없음"}
                      </Link>
                    ) : (
                      paper.title || "제목 없음"
                    )
                  }
                </Typography>
              }
              secondary={
                <>
                  <Typography component="span" variant="body2" color="text.primary">
                    저자: {paper.authors || "정보 없음"}
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
                      {paper.abstract || "초록 정보 없음"}
                    </Typography>
                    {paper.abstract && (
                      <Button 
                        onClick={() => toggleAbstract(index)} 
                        sx={{ mt: 1, p: 0, minWidth: 'unset' }}
                      >
                        {expandedAbstracts[index] ? '접기' : '더보기'}
                      </Button>
                    )}
                  </Box>
                  {/* 유사도 수치 표시 제거됨 */}
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