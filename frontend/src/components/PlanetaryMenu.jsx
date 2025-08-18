// import React from 'react';
// import { Planet } from 'react-planet';
// import MenuIcon from '@mui/icons-material/Menu';
// import MailIcon from '@mui/icons-material/Mail';
// import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
// import MapIcon from '@mui/icons-material/Map';
// import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';
// import DescriptionIcon from '@mui/icons-material/Description';
// import { styled } from '@mui/system';

// // Styled satellite item
// const SatelliteItem = styled('div')(({ theme }) => ({
//   width: 50,
//   height: 50,
//   borderRadius: '50%',
//   backgroundColor: theme.palette.primary.main,
//   display: 'flex',
//   justifyContent: 'center',
//   alignItems: 'center',
//   color: theme.palette.primary.contrastText,
//   cursor: 'pointer',
//   '&:hover': {
//     backgroundColor: theme.palette.primary.dark,
//   },
// }));

// // SWOT submenu
// const SWOTMenu = () => (
//   <Planet
//     centerContent={<SatelliteItem><MailIcon /></SatelliteItem>}
//     hideOrbit
//     autoClose
//     orbitRadius={50}
//     bounceOnClose
//   >
//     <SatelliteItem onClick={() => alert('Strengths Detailed')}>
//       <MailIcon />
//     </SatelliteItem>
//     <SatelliteItem onClick={() => alert('Weaknesses Detailed')}>
//       <HelpOutlineIcon />
//     </SatelliteItem>
//     <SatelliteItem onClick={() => alert('Opportunities Detailed')}>
//       <MapIcon />
//     </SatelliteItem>
//     <SatelliteItem onClick={() => alert('Threats Detailed')}>
//       <MapIcon />
//     </SatelliteItem>
//   </Planet>
// );

// // Pitch Support submenu
// const PitchSupportMenu = () => (
//   <Planet
//     centerContent={<SatelliteItem><MonetizationOnIcon /></SatelliteItem>}
//     hideOrbit
//     autoClose
//     orbitRadius={100}
//     bounceOnClose
//   >
//     <SatelliteItem onClick={() => alert('Elevator Pitch')}>
//       <DescriptionIcon />
//     </SatelliteItem>
//     <SatelliteItem onClick={() => alert('Monetization Strategies')}>
//       <MonetizationOnIcon />
//     </SatelliteItem>
//     <SatelliteItem onClick={() => alert('PitchDeck Structure')}>
//       <DescriptionIcon />
//     </SatelliteItem>
//   </Planet>
// );

// // Main menu component
// const PlanetaryMenu = () => (
//   <Planet
//     centerContent={<SatelliteItem><MenuIcon /></SatelliteItem>}
//     open
//     autoClose
//     orbitRadius={150}
//     bounceOnClose
//   >
//     <SWOTMenu />
//     <PitchSupportMenu />
//   </Planet>
// );

// export default PlanetaryMenu;
