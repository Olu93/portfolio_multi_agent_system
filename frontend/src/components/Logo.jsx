import { Typography } from "@mui/material";

export function FullLogo() {
  return (
    <Typography
      variant="h6"
      noWrap
      sx={{
        mr: 2,
        display: { xs: "none", md: "flex" },
        fontFamily: "monospace",
        fontWeight: 700,
        letterSpacing: ".3rem",
        color: "inherit",
        textDecoration: "none",
      }}
    >
      AILUV/IO
    </Typography>
  );
}
export function ShortLogo() {
  return (
    <Typography
      variant="h6"
      noWrap
      sx={{
        mr: 2,
        display: { xs: "none", md: "flex" },
        fontFamily: "monospace",
        fontWeight: 700,
        letterSpacing: ".3rem",
        color: "inherit",
        textDecoration: "none",
      }}
    >
      A/IO
    </Typography>
  );
}
