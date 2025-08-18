import React, { useState, useEffect } from 'react';
import { Box, CircularProgress, Typography, Card, CardContent, List, CardHeader, IconButton, CardActions, Tab, Tabs } from '@mui/material';
import Grid from '@mui/material/Grid2';
import axios from 'axios';
import CustomListComponent from './CustomListComponent';
import Lightbulp from '@mui/icons-material/LightbulbSharp';
import { styled } from '@mui/material/styles';


function SWOTProfile({ responseData }) {
    const [expanded, setExpanded] = useState(false);
    const [selectedTab, setSelectedTab] = useState(0);

    const {text, swot} = responseData;

    const handleTabChange = (event, newValue) => {
        setSelectedTab(newValue);
    };

    return (
        <Grid>
            <Card>
                <CardHeader title={<Typography variant="h5">SWOT Analysis</Typography>} avatar={<Lightbulp />} />

                <CardContent>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                        {text}
                    </Typography>

                    <Tabs value={selectedTab}
                        onChange={handleTabChange} sx={{ mt: 2 }}
                        variant="scrollable"
                        scrollButtons
                        allowScrollButtonsMobile
                    >
                        <Tab label="Strength" />
                        <Tab label="Weaknesses" />
                        <Tab label="Opportunities" />
                        <Tab label="Threats" />
                    </Tabs>

                    {/* Tab Panels */}
                    {selectedTab === 0 && (
                        <Box sx={{ mt: 2 }}>
                            <List id="Strengths">
                                {swot.strengths.map((strength, index) => <CustomListComponent key={index} title={strength.title} description={strength.description} />)}
                            </List>
                        </Box>
                    )}
                    {selectedTab === 1 && (
                        <Box sx={{ mt: 2 }}>
                            <List id="Weaknesses">
                                {swot.weaknesses.map((weakness, index) => <CustomListComponent key={index} title={weakness.title} description={weakness.description} />)}
                            </List>
                        </Box>
                    )}
                    {selectedTab === 2 && (
                        <Box sx={{ mt: 2 }}>
                            <List id="Opportunities">
                                {swot.opportunities.map((opportunity, index) => <CustomListComponent key={index} title={opportunity.title} description={opportunity.description} />)}
                            </List>
                        </Box>
                    )}
                    {selectedTab === 3 && (
                        <Box sx={{ mt: 2 }}>
                            <List id="Threats">
                                {swot.threats.map((threat, index) => <CustomListComponent key={index} title={threat.title} description={threat.description} />)}
                            </List>
                        </Box>
                    )}
                </CardContent>
            </Card>
        </Grid>
    );
}

export default SWOTProfile;
