import { ListItem, ListItemIcon, ListItemText } from "@mui/material";
import SendIcon from '@mui/icons-material/Send';

export default function CustomListComponent({ title, description }) {
    let formattedDescription = description;

    if (!description || (Array.isArray(description) && description.length === 0)) {
        formattedDescription = `${title} was not provided.`;
    } else if (Array.isArray(description)) {
        formattedDescription = description
            .map(item => item.endsWith('.') ? item.slice(0, -1) : item)
            .join('. ') + '.';
    }

    return (
        <ListItem>
            <ListItemIcon>
                <SendIcon />
            </ListItemIcon>
            <ListItemText
                primary={title}
                secondary={formattedDescription}
            />
        </ListItem>
    );
}