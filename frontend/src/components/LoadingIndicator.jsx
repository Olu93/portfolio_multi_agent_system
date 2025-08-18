import { Box } from "@mui/material";
import CircularProgress from "@mui/material/CircularProgress";

export default function LoadingIndicator() {
  return (
    <Box display="flex" justifyContent="center">
      <CircularProgress />
    </Box>
  );
}