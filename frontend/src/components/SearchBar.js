import React, { useState } from 'react';
import { TextField, Button, Box } from '@mui/material';

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', justifyContent: 'center' }}>
      <TextField
        fullWidth
        variant="outlined"
        label="어떤 지식을 알고 싶으신가요?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        sx={{ mr: 2, maxWidth: '600px' }}
      />
      <Button type="submit" variant="contained">
        Search
      </Button>
    </Box>
  );
}

export default SearchBar;