import React, { useState, useRef, useEffect } from 'react';
// import {
//   MainContainer,
//   ChatContainer,
//   MessageList,
//   Message,
//   MessageInput,
//   TypingIndicator,
//   ConversationHeader,
//   MessageSeparator,
// } from '@chatscope/chat-ui-kit-react';
// import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
// import './App.css';
import Copilot from './pages/Copilot.jsx';

// Mock data for supervisor agent only
const mockConversations = [
  {
    id: 1,
    message: "Hello! I'm the Supervisor Agent. I can help coordinate tasks between different specialized agents. What would you like me to help you with?",
    sender: 'supervisor',
    direction: 'incoming',
    timestamp: new Date(Date.now() - 60000)
  }
];

function App() {
  const [conversations, setConversations] = useState(mockConversations);
  const [isTyping, setIsTyping] = useState(false);
  const [messageInputValue, setMessageInputValue] = useState('');
  const messageListRef = useRef();

  const handleSendMessage = (message) => {
    if (message.trim() === '') return;

    const newMessage = {
      id: Date.now(),
      message: message,
      sender: 'user',
      direction: 'outgoing',
      timestamp: new Date()
    };

    // Add user message
    setConversations(prev => [...prev, newMessage]);

    setMessageInputValue('');
    setIsTyping(true);

    // Simulate supervisor response after a delay
    setTimeout(() => {
      const supervisorResponse = generateSupervisorResponse(message);
      const responseMessage = {
        id: Date.now() + 1,
        message: supervisorResponse,
        sender: 'supervisor',
        direction: 'incoming',
        timestamp: new Date()
      };

      setConversations(prev => [...prev, responseMessage]);
      setIsTyping(false);
    }, 1000 + Math.random() * 2000); // Random delay between 1-3 seconds
  };

  const generateSupervisorResponse = (userMessage) => {
    const responses = [
      "I'll coordinate with the appropriate agents to help you with that. Let me analyze your request and delegate it to the right specialist.",
      "That's a great question! I can help orchestrate a solution using our specialized agents. Let me break this down into manageable tasks.",
      "I understand your request. Let me coordinate with the relevant agents to provide you with a comprehensive solution.",
      "Excellent! I'll supervise the workflow and ensure all agents work together efficiently to complete your task.",
      "I can help coordinate this request. Let me identify which specialized agents would be best suited for your needs.",
      "That's a complex task that requires coordination. I'll orchestrate the workflow between our specialized agents to get you the best results."
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  };

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollToBottom();
    }
  }, [conversations]);

  return (
    <Copilot />
  );
}

export default App;
