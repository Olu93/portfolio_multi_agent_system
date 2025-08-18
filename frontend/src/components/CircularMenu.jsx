import MailIcon from "@mui/icons-material/Mail";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import MapIcon from "@mui/icons-material/Map";
import AddIcon from "@mui/icons-material/AddCircle";
import RemoveIcon from "@mui/icons-material/RemoveCircle";
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShortTextIcon from '@mui/icons-material/ShortText';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';

// Import the circular menu
import {
  CircleMenu,
  CircleMenuItem,
  TooltipPlacement,
} from "react-circular-menu";

export const CicularMenuComponent = (props) => {
  return (
    <CircleMenu
      startAngle={-90}
      rotationAngle={360}
      itemSize={2}
      radius={5}
      /**
       * rotationAngleInclusive (default true)
       * Whether to include the ending angle in rotation because an
       * item at 360deg is the same as an item at 0deg if inclusive.
       * Leave this prop for angles other than 360deg unless otherwise desired.
       */
      rotationAngleInclusive={false}
    >
      <CircleMenuItem
        onClick={() => alert("Clicked the item")}
        tooltip="Strengths Detailed"
        tooltipPlacement={TooltipPlacement.Right}
      >
        <AddIcon />
      </CircleMenuItem>
      <CircleMenuItem tooltip="Weaknesses Detailed">
        <RemoveIcon />
      </CircleMenuItem>
      <CircleMenuItem tooltip="Opportunities Detailed">
        <TrendingUpIcon />
      </CircleMenuItem>
      <CircleMenuItem tooltip="Threats Detailed">
        <TrendingDownIcon />
      </CircleMenuItem>
      <CircleMenuItem tooltip="Elevator Pitch">
        <ShortTextIcon />
      </CircleMenuItem>
      <CircleMenuItem tooltip="Monetization Strategies">
        <AttachMoneyIcon />
      </CircleMenuItem>
      <CircleMenuItem tooltip="PitchDeck Structure">
        <FormatListBulletedIcon />
      </CircleMenuItem>
      {/* <CircleMenuItem tooltip="Info">
        <InfoIcon />
      </CircleMenuItem> */}
    </CircleMenu>
  );
};