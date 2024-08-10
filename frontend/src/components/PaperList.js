import React, { useState } from 'react';
import { Typography, Paper, List, ListItem, ListItemText, Link, Box, Button } from '@mui/material';

const PaperList = ({ papers, gptAnswer }) => {
  const [expandedAbstracts, setExpandedAbstracts] = useState({});
  const [showAllResults] = useState(false);

  const toggleAbstract = (index) => {
    setExpandedAbstracts(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  // GPT의 답변이 "nodata"인 경우
  if (gptAnswer.includes("nodata")) {
    return <Typography></Typography>;
  }

  // 논문 리스트가 비어있는 경우
  if (!papers || papers.length === 0) {
    return <Typography></Typography>;
  }

  // 유사도에 따른 논문 필터링
  const validPapers = showAllResults ? papers : papers.filter(paper => paper.similarity >= 0.7);

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ color: 'primary.main' }}>논문 출처</Typography>
      </Box>
      {validPapers.length > 0 ? (
        <List>
          {validPapers.map((paper, index) => (
            <ListItem key={index} alignItems="flex-start" sx={{ px: 0 }}>
              <ListItemText
                primary={
                  paper.url && paper.url !== "정보 없음" ? (
                    <Link href={paper.url} target="_blank" rel="noopener noreferrer" color="primary">
                      {`${index + 1}. ${paper.title || "제목 없음"}`}
                    </Link>
                  ) : (
                    `${index + 1}. ${paper.title || "제목 없음"}`
                  )
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
                    <Typography variant="body2" fontWeight="bold" color="text.secondary" sx={{ mt: 1 }}>
                      유사도: {paper.similarity != null ? `${(paper.similarity * 100).toFixed(2)}%` : '정보 없음'}
                    </Typography>
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      ) : (
        <Typography>질문과 유사한 논문이 없습니다.</Typography>
      )}
    </Paper>
  );
}

export default PaperList;