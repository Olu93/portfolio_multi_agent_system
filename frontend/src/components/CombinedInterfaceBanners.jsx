import React from 'react';
import { Box, Typography, TextField, Button } from '@mui/material';

const ChatInterfaceBanner = () => {
  return (
    <Box
      sx={{
        padding: 2,
        borderRadius: 1,
        marginBottom: 2,
      }}
    >
      <Typography variant="h6" gutterBottom>
        Welcome to the Chat Interface!
      </Typography>
      <Typography variant="body1" gutterBottom>
        Explore the master agent. You can:
      </Typography>
      <ul>
        <li>
          <Typography variant="body2">
            <strong>Research the web:</strong> Search the web for information.
          </Typography>
        </li>
        <li>
          <Typography variant="body2">
            <strong>Schedule a meeting or send an email:</strong> Schedule a meeting or send an email.
          </Typography>
        </li>
        <li>
          <Typography variant="body2">
            <strong>Get current stock prices:</strong> Get current stock prices of for instance Apple, Google, etc.
          </Typography>
        </li>
      </ul>
      <Typography variant="body1" gutterBottom>
        Start simple, like "Tell me about apple's current stock price." Need help? Type "What can I do?"
      </Typography>
    </Box>
  );
};

const SemanticSearchBanner = () => {
  return (
    <Box
      sx={{
        padding: 2,
        borderRadius: 1,
        marginBottom: 1,
      }}
    >
      <Typography variant="h6" gutterBottom>
        Welcome to Startup Search!
      </Typography>
      <Typography variant="body1" gutterBottom>
        Find startups quickly and easily with our semantic search. Simply type your query to:
      </Typography>
      <ul>
        <li>
          <Typography variant="body2">
            <strong>Discover Startups:</strong> Search by industry, technology, or trends.
          </Typography>
        </li>
        <li>
          <Typography variant="body2">
            <strong>Explore Innovations:</strong> Learn about groundbreaking ideas and business models.
          </Typography>
        </li>
        <li>
          <Typography variant="body2">
            <strong>Get Insights:</strong> Find detailed information about founders, funding, and more.
          </Typography>
        </li>
      </ul>
    </Box>
  );
};

const AnalyzerBanner = () => {
  return (
    <Box
      sx={{
        padding: 2,
        borderRadius: 1,
        marginBottom: 2,
      }}
    >
      <Typography variant="h6" gutterBottom>
        Welcome to the Pitch Analyzer!
      </Typography>
      <Typography variant="body1" gutterBottom>
        Upload a pitch and gain valuable insights instantly. Here's what you can do:
      </Typography>
      <ul>
        <li>
          <Typography variant="body2">
            <strong>Idea Summary:</strong> Get a concise summary of the pitch.
          </Typography>
        </li>
        <li>
          <Typography variant="body2">
            <strong>Strengths & Threats:</strong> Understand key opportunities and risks.
          </Typography>
        </li>
        <li>
          <Typography variant="body2">
            <strong>Competitor Analysis:</strong> See how it stacks up against similar ideas.
          </Typography>
        </li>
      </ul>
    </Box>
  );
};

const FounderAnalyzerBanner = () => {
  return (
    <Box>
      <Typography variant="h6">Pitchdeck Analyzer</Typography>
      <Typography variant="body2">
        Upload your pitch deck to get a SWOT analysis and detailed feedback on your business idea.
      </Typography>
    </Box>
  );
};

export { ChatInterfaceBanner, SemanticSearchBanner, AnalyzerBanner, FounderAnalyzerBanner };
