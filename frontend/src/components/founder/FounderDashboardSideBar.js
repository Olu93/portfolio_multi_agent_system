import React from "react";
import {
  List,
  Divider,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import ChatIcon from "@mui/icons-material/Chat";
import SearchIcon from "@mui/icons-material/Search";
import SettingsIcon from "@mui/icons-material/Settings";
import InfoIcon from "@mui/icons-material/Info";
import { ShortLogo } from "../Logo";
import Box from "@mui/material/Box";
import HomeIcon from "@mui/icons-material/Home";
import DashboardSideBar, { MenuDrawerHeader, MenuItem } from "../DashboardSideBar";

const drawerWidth = 80;


const FounderDashboardSideBar = () => {
  const theme = useTheme();
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const drawer = (
    <Box>
      <MenuDrawerHeader>
        <ShortLogo />
      </MenuDrawerHeader>
      <Divider />
      
      <List>
        <MenuItem path="/founder" icon={<HomeIcon />} title="Home" />
        <MenuItem path="/founder/analyzer" icon={<QueryStatsIcon />} title="Analyzer" />
      </List>
    </Box>
  );

  return (
    <DashboardSideBar>{drawer}</DashboardSideBar>
  );
};

export default FounderDashboardSideBar;
