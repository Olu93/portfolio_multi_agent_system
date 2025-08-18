import React, { useState } from "react";
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Paper,
  InputAdornment,
  useMediaQuery,
  Drawer,
  AppBar,
  Toolbar,
  Alert,
} from "@mui/material";
import Grid from "@mui/material/Grid2";
import { styled } from "@mui/system";
import SendIcon from "@mui/icons-material/Send";
import SearchIcon from "@mui/icons-material/Search";
import MenuIcon from "@mui/icons-material/Menu";
import axios from "axios";
import Markdown from "react-markdown";
import { ChatInterfaceBanner } from "../components/CombinedInterfaceBanners";
import { A2AClient } from "@a2a-js/sdk/client";
import { v4 as uuidv4 } from "uuid";

const StyledContainer = styled(Box)(({ theme }) => ({
  display: "flex",
  height: "100vh",
  color: "#ffffff",
}));

const StyledMessageContainer = styled(Box)({
  flex: 1,
  overflowY: "auto",
  padding: "20px",
});

const StyledMessage = styled(Paper)(({ isOwn }) => ({
  padding: "10px 15px",
  margin: "8px 0",
  maxWidth: isOwn ? "70%" : "100%",
  width: "fit-content",
  background: isOwn ? "#0d47a1" : "#121212",
  color: "#ffffff",
  borderRadius: "12px",
  alignSelf: isOwn ? "flex-end" : "flex-start",
  boxShadow: !isOwn && "none",
}));

const StyledInputArea = styled(Box)({
  padding: "20px",
});

const StyledSearchBox = styled(TextField)({
  "& .MuiOutlinedInput-root": {
    color: "#ffffff",
    "& fieldset": {
      borderColor: "#424242",
    },
    "&:hover fieldset": {
      borderColor: "#666666",
    },
    "&.Mui-focused fieldset": {
      borderColor: "#0d47a1",
    },
  },
});

const ChatUI = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const isMobile = useMediaQuery("(max-width:600px)");

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMessageSend = async () => {
    if (message.trim()) {
      const userMessage = {
        id: messages.length + 1,
        text: message,
        isOwn: true,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages([...messages, userMessage]);
      setMessage("");

      try {
        const response = await fetch("http://127.0.0.1:8000/pitchdeck/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: message,
            history: messages.map(
              (msg) => `${msg.isOwn ? "User" : "AI"}:${msg.text}`
            ),
          }),
        });

        if (!response.body) {
          throw new Error(
            "ReadableStream is not supported in this environment."
          );
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let result = "";

        // Add a placeholder message for the bot's response
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: prevMessages.length + 2,
            text: "",
            isOwn: false,
            timestamp: new Date().toLocaleTimeString(),
          }, //Placeholder
        ]);
        let messageIndex = messages.length + 1;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          result += decoder.decode(value, { stream: true });

          // Update the LAST message in the array
          setMessages((prevMessages) => {
            const updatedMessages = [...prevMessages];
            updatedMessages[messageIndex] = {
              ...updatedMessages[messageIndex],
              text: result,
            };
            return updatedMessages;
          });
        }
      } catch (error) {
        console.error("Error sending message:", error);
        // Handle error, e.g., by updating the last message with an error message
        setMessages((prevMessages) => {
          const updatedMessages = [...prevMessages];
          updatedMessages[updatedMessages.length - 1] = {
            ...updatedMessages[updatedMessages.length - 1],
            text: "Error fetching response.",
          };
          return updatedMessages;
        });
      }
    }
  };


  const handleMessageSend2 = async () => {
    if (message.trim()) {
      const userMessage = {
        id: messages.length + 1,
        text: message,
        isOwn: true,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages([...messages, userMessage]);
      setMessage("");

      try {
        // Initialize A2A client
        const client = new A2AClient("http://localhost:41241"); // Replace with your server URL
        const messageId = uuidv4();
        let taskId;

        // 1. Send a message to the agent
        const sendParams = {
          message: {
            messageId: messageId,
            role: "user",
            parts: [{ kind: "text", text: message }],
            kind: "message",
          },
          configuration: {
            blocking: true,
            acceptedOutputModes: ["text/plain"],
          },
        };

        const sendResponse = await client.sendMessage(sendParams);

        if (sendResponse.error) {
          console.error("Error sending message:", sendResponse.error);
          // Add error message to chat
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              id: prevMessages.length + 2,
              text: "Error sending message to A2A server.",
              isOwn: false,
              timestamp: new Date().toLocaleTimeString(),
            },
          ]);
          return;
        }

        // On success, check if result is a Task or a Message
        const result = sendResponse.result;

        if (result.kind === "task") {
          // The agent created a task
          console.log("Send Message Result (Task):", result);
          taskId = result.id; // Save the task ID for the next call
        } else if (result.kind === "message") {
          // The agent responded with a direct message
          console.log("Send Message Result (Direct Message):", result);
          // Add the AI response to chat
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              id: prevMessages.length + 2,
              text: result.parts[0]?.text || "AI response received",
              isOwn: false,
              timestamp: new Date().toLocaleTimeString(),
            },
          ]);
        }

        // 2. If a task was created, get its status
        if (taskId) {
          const getParams = { id: taskId };
          const getResponse = await client.getTask(getParams);

          if (getResponse.error) {
            console.error(`Error getting task ${taskId}:`, getResponse.error);
            // Add error message to chat
            setMessages((prevMessages) => [
              ...prevMessages,
              {
                id: prevMessages.length + 2,
                text: `Error getting task status: ${getResponse.error.message}`,
                isOwn: false,
                timestamp: new Date().toLocaleTimeString(),
              },
            ]);
            return;
          }

          const getTaskResult = getResponse.result;
          console.log("Get Task Result:", getTaskResult);
          
          // Add the task result to chat
          if (getTaskResult.status === "completed" && getTaskResult.output) {
            setMessages((prevMessages) => [
              ...prevMessages,
              {
                id: prevMessages.length + 2,
                text: getTaskResult.output[0]?.text || "Task completed",
                isOwn: false,
                timestamp: new Date().toLocaleTimeString(),
              },
            ]);
          }
        }
      } catch (error) {
        console.error("A2A Client Communication Error:", error);
        // Handle error by adding error message to chat
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: prevMessages.length + 2,
            text: `Communication error: ${error.message}`,
            isOwn: false,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      }
    }
  };

  return (
    <StyledContainer>
      <Grid width={"100%"} size={12} id="test">
        <Grid
          item
          size={12}
          md={3}
          direction={"row"}
          alignItems={"stretch"}
          justifyContent={"center"}
        >
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            sx={{ height: "100%" }}
          >
            <Typography
              variant="h2"
              sx={{ fontFamily: "monospace", textDecoration: "none" }}
            >
              A/I Copilot
            </Typography>

            <Alert severity="info" sx={{ marginTop: 5 }}>
              <ChatInterfaceBanner />
            </Alert>
          </Box>
        </Grid>
        <Grid item size={12} md={9}>
          {/* <Box display="flex" flexDirection="column" height="100%"> */}
          <StyledMessageContainer>
            <Box display="flex" flexDirection="column" gap={1}>
              {messages.map((msg) => (
                <Box
                  key={msg.id}
                  display="flex"
                  flexDirection="column"
                  alignItems={msg.isOwn ? "flex-end" : "flex-start"}
                >
                  <StyledMessage isOwn={msg.isOwn}>
                    <Markdown>{msg.text}</Markdown>
                    <Typography variant="caption" sx={{ color: "#cccccc" }}>
                      {msg.timestamp}
                    </Typography>
                  </StyledMessage>
                </Box>
              ))}
            </Box>
          </StyledMessageContainer>

          <StyledInputArea>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type a message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleMessageSend()}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={handleMessageSend}
                      sx={{ color: "#ffffff" }}
                    >
                      <SendIcon />
                    </IconButton>
                    <IconButton
                      onClick={handleMessageSend2}
                      sx={{ color: "#0d47a1", marginLeft: 1 }}
                      title="Send via A2A Server"
                    >
                      <SendIcon />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                "& .MuiOutlinedInput-root": {
                  color: "#ffffff",
                  "& fieldset": { borderColor: "#424242" },
                  "&:hover fieldset": { borderColor: "#666666" },
                  "&.Mui-focused fieldset": { borderColor: "#0d47a1" },
                },
              }}
            />
          </StyledInputArea>
          {/* </Box> */}
        </Grid>
      </Grid>
    </StyledContainer>
  );
};

export default ChatUI;
