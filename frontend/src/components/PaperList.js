import React from 'react';
import { Typography, Paper, List, ListItem, ListItemText, Link } from '@mui/material';

function PaperList({ papers }) {
  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: '16px' }}>
      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>논문 리스트</Typography>
      <List>
        {papers.map((paper, index) => (
          <ListItem key={index} alignItems="flex-start" sx={{ px: 0 }}>
            <ListItemText
              primary={`${index + 1}. ${paper.title}`}
              secondary={
                <>
                  <Typography component="span" variant="body2" color="text.primary">
                    저자: {paper.authors}
                  </Typography>
                  <br />
                  {paper.url && (
                    <Link href={paper.url} target="_blank" rel="noopener noreferrer" color="primary">
                      논문 보기
                    </Link>
                  )}
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