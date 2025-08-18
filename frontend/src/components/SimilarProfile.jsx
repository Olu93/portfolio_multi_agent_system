import React, { useState, useEffect } from "react";
import {
  Box,
  CircularProgress,
  Typography,
  Card,
  CardContent,
  List,
  CardHeader,
  IconButton,
  CardActions,
  Tab,
  Tabs,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import Grid from "@mui/material/Grid2";
import axios from "axios";
import CustomListComponent from "./CustomListComponent";
import Lightbulp from "@mui/icons-material/LightbulbSharp";
import { styled } from "@mui/material/styles";
import SendIcon from '@mui/icons-material/Send';


function SimilarProfile({ responseData }) {
  const [expanded, setExpanded] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  return (
    <Grid>
      <Card>
        <CardHeader
          title={<Typography variant="h5">Similar Startups</Typography>}
          avatar={<Lightbulp />}
        />

        <CardContent>
          <Box sx={{ mt: 2 }}>
            <List id="Strengths">
              {responseData.map((company, index) => (
                <ListItem>
                  <Typography variant="body2" sx={{ color: "text.secondary" }}>
                    {company.document_data.short_summary}
                  </Typography>
                  <ListItemIcon>
                    <SendIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary={company.document_data.company_name}
                    secondary={company.document_data.similarity_reason}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </CardContent>
      </Card>
    </Grid>
  );
}

export default SimilarProfile;
