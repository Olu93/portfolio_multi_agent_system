import React, { useState, useEffect } from 'react';
import { Box, CircularProgress, Typography, Card, CardContent, List, CardHeader, IconButton, CardActions, Tab, Tabs } from '@mui/material';
import Grid from '@mui/material/Grid2';
import axios from 'axios';
import CustomListComponent from './CustomListComponent';
import Lightbulp from '@mui/icons-material/LightbulbSharp';
import { styled } from '@mui/material/styles';

const ExpandMore = styled((props) => {
    const { expand, ...other } = props;
    return <IconButton {...other} />;
})(({ theme, expand }) => ({
    marginLeft: 'auto',
    transition: theme.transitions.create('transform', {
        duration: theme.transitions.duration.shortest,
    }),
    transform: !expand ? 'rotate(0deg)' : 'rotate(180deg)',
}));

function CompanyProfile({ responseData, withCompetitorAnalysis }) {
    const [expanded, setExpanded] = useState(false);
    const [competitorsData, setCompetitorsData] = useState(null);
    const [analysisLoading, setAnalysisLoading] = useState(true);
    const [selectedTab, setSelectedTab] = useState(0);

    const pitch_deck_content = responseData.pitch_deck_content;

    useEffect(() => {
        const fetchCompetitorData = async () => {
            try {
                const competitorResponse = await axios.post('http://127.0.0.1:8000/pitchdeck/competitors', {
                    company_name: responseData.company_name,
                    executive_summary: responseData.executive_summary,
                    key_highlights: responseData.key_highlights,
                    potential_risks: responseData.potential_risks,
                    recommendations: responseData.recommendations,
                    pitch_deck_content: responseData.pitch_deck_content
                }, {
                    headers: {
                        'accept': 'application/json',
                        'Content-Type': 'application/json',
                    },
                });

                setCompetitorsData(competitorResponse.data);
                console.log("Competitor Analysis successful:", competitorResponse.data);
            } catch (error) {
                console.error("Error fetching competitor analysis:", error);
            } finally {
                setAnalysisLoading(false);
            }
        };
        if (withCompetitorAnalysis){
            fetchCompetitorData();
        }
    }, [responseData]);

    const handleExpandClick = () => {
        setExpanded(!expanded);
    };

    const handleTabChange = (event, newValue) => {
        setSelectedTab(newValue);
    };

    return (
        <Grid size={6}>
            <Card>
                <CardHeader title={<Typography variant="h5">{responseData.company_name}</Typography>} avatar={<Lightbulp />} />

                <CardContent>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                        {responseData.executive_summary}
                    </Typography>

                    <Tabs value={selectedTab}
                        onChange={handleTabChange} sx={{ mt: 2 }}
                        variant="scrollable"
                        scrollButtons
                        allowScrollButtonsMobile
                    >
                        <Tab label="Assessment" />
                        <Tab label={`PitchDeck`} />
                        {withCompetitorAnalysis && <Tab label={"Competitors "} icon={(analysisLoading ? <CircularProgress size={24} /> : '')} iconPosition="end" disabled={analysisLoading} />}
                        {withCompetitorAnalysis && <Tab label={"Partners "} icon={(analysisLoading ? <CircularProgress size={24} /> : '')} iconPosition="end" disabled={analysisLoading} />}
                        {withCompetitorAnalysis && <Tab label={"Remarks "} icon={(analysisLoading ? <CircularProgress size={24} /> : '')} iconPosition="end" disabled={analysisLoading} />}
                    </Tabs>

                    {/* Tab Panels */}
                    {selectedTab === 0 && (
                        <Box sx={{ mt: 2 }}>
                            <List id="Assessment">
                                <CustomListComponent title={"Key Highlights"} description={responseData.key_highlights} />
                                <CustomListComponent title={"Potential Risks"} description={responseData.potential_risks} />
                                <CustomListComponent title={"Recommendations"} description={responseData.recommendations} />
                            </List>
                        </Box>
                    )}

                    {selectedTab === 1 && (
                        <Box sx={{ mt: 2 }}>
                            <List id="PitchDeck Content">
                                <CustomListComponent title={"Mission & Vision"} description={pitch_deck_content?.mission_vision} />
                                <CustomListComponent title={"Problem Statement"} description={pitch_deck_content?.problem_statement} />
                                <CustomListComponent title={"Market Size"} description={pitch_deck_content?.market_size} />
                                <CustomListComponent title={"Product"} description={pitch_deck_content?.product_uniqueness} />
                                <CustomListComponent title={"Traction & Revenue"} description={pitch_deck_content?.traction_revenue} />
                                <CustomListComponent title={"Team"} description={pitch_deck_content?.team_experience} />
                            </List>
                        </Box>
                    )}

                    {selectedTab === 2 && competitorsData && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="h6">Competitors</Typography>
                            <List>
                                {competitorsData.competitors.map((competitor, index) => (
                                    <CustomListComponent key={index} title={competitor.name} description={competitor.description} />
                                ))}
                            </List>
                        </Box>
                    )}

                    {selectedTab === 3 && competitorsData && competitorsData.partners && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="h6">Potential Partners</Typography>
                            <List>
                                {competitorsData.partners.map((partner, index) => (
                                    <CustomListComponent key={index} title={partner.name} description={partner.description} />
                                ))}
                            </List>
                        </Box>
                    )}

                    {selectedTab === 4 && competitorsData && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="h6">Competitive Advantages & Additional Remarks</Typography>
                            {competitorsData.competitive_advantage && (
                                <CustomListComponent title={"Competitive Advantage"} description={competitorsData.competitive_advantage} />
                            )}
                            {competitorsData.additional_remarks && (
                                <CustomListComponent title={"Additional Remarks"} description={competitorsData.additional_remarks} />
                            )}
                        </Box>
                    )}
                </CardContent>
                
                <CardActions disableSpacing>
                    <Typography variant="caption" sx={{ marginLeft: "auto", marginRight: 2 }}>
                        Submitted on {new Date(responseData.storage_datetime).toLocaleString()}
                    </Typography>
                </CardActions>
            </Card>
        </Grid>
    );
}

export default CompanyProfile;
