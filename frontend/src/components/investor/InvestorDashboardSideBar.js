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


const InvestorDashboardSideBar = () => {
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
        <MenuItem path="/investor" icon={<HomeIcon />} title="Home" />
        <MenuItem path="/investor/analyzer" icon={<QueryStatsIcon />} title="Analyzer" />
        <MenuItem path="/investor/copilot" icon={<ChatIcon />} title="Copilot" />
        <MenuItem path="/investor/browser" icon={<SearchIcon />} title="Browse" />
        <MenuItem path="/investor/settings" icon={<SettingsIcon />} title="Settings" />
        <MenuItem path="/investor/about" icon={<InfoIcon />} title="About" />
      </List>
    </Box>
  );

  return (
    <DashboardSideBar>{drawer}</DashboardSideBar>
  );
};

export default InvestorDashboardSideBar;
