// import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
// import './App.css'

// function App() {
//   const [count, setCount] = useState(0)

//   return (
//     <>
//       <div>
//         <a href="https://vite.dev" target="_blank">
//           <img src={viteLogo} className="logo" alt="Vite logo" />
//         </a>
//         <a href="https://react.dev" target="_blank">
//           <img src={reactLogo} className="logo react" alt="React logo" />
//         </a>
//       </div>
//       <h1>Vite + React</h1>
//       <div className="card">
//         <button onClick={() => setCount((count) => count + 1)}>
//           count is {count}
//         </button>
//         <p>
//           Edit <code>src/App.jsx</code> and save to test HMR
//         </p>
//       </div>
//       <p className="read-the-docs">
//         Click on the Vite and React logos to learn more
//       </p>
//     </>
//   )
// }

// export default App

import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  // const [messages, setMessages] = useState([]); // Stores the chat history
  const [messages, setMessages] = useState([
    { role: 'ai', content: "Hello! I'm your Dental Assistant AI. How can I help you today?" }
  ]);
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const socket = useRef(null);
  const scrollRef = useRef(null);
  const IS_OFFLINE = true;

  // 1. Setup WebSocket Connection
  useEffect(() => {
    if (IS_OFFLINE) return; // Don't try to connect if we are testing locally

    // Replace with your teammate's actual IP/URL (e.g., ws://192.168.1.10:8000/ws)
    socket.current = new WebSocket('ws://localhost:8000/ws/chat');

    socket.current.onopen = () => console.log("✅ Connected to AI Backend");
    
    socket.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Standard streaming pattern: backend sends { "type": "token", "content": "Hello" }
        if (data.type === 'token') {
          updateLastAiMessage(data.content);
        } 
        
        // Backend sends { "type": "end" } when the sentence is finished
        if (data.type === 'end') {
          setIsTyping(false);
        }
      } catch (err) {
        console.error("Format error:", err);
      }
    };

    socket.current.onclose = () => {
      console.log("❌ Disconnected");
      setIsTyping(false);
    };

    return () => socket.current?.close();
  }, [IS_OFFLINE]); // Re-runs if you toggle the mode

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const updateLastAiMessage = (token) => {
    setMessages((prev) => {
      const lastMsg = prev[prev.length - 1];
      
      // Check if the last message is from the AI
      if (lastMsg && lastMsg.role === 'ai') {
        // Create a copy of the message list
        const newMessages = [...prev];
        // Update ONLY the content of the last message
        newMessages[newMessages.length - 1] = {
          ...lastMsg,
          content: lastMsg.content + token
        };
        return newMessages;
      } else {
        // If the last message wasn't AI, start a new AI bubble
        return [...prev, { role: 'ai', content: token }];
      }
    });
  };

  const handleSend = () => {
    if (!input.trim()) return;

    // 1. Add User Message to UI
    const userMsg = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    const currentInput = input; // Store it before clearing
    setInput('');
    setIsTyping(true);

    if (IS_OFFLINE) {
      // --- DUMMY MODE ---
      runDummyResponse();
    } else {
      // --- LIVE MODE (Websocket) ---
      if (socket.current && socket.current.readyState === WebSocket.OPEN) {
        socket.current.send(JSON.stringify({ 
          message: currentInput, 
          session_id: currentChatId || "new_session" 
        }));
      } else {
        // Failure Handling (Requirement 6)
        setMessages(prev => [...prev, { role: 'ai', content: "⚠️ Error: Connection lost. Please refresh." }]);
        setIsTyping(false);
      }
    }
  };

  // Separate the Dummy Logic into its own function to keep handleSend clean
  const runDummyResponse = () => {
    setTimeout(() => {
      const dummyText = "I am currently in Offline Mode. Change 'IS_OFFLINE' to false to connect to the backend!";
      const words = dummyText.split(" ");
      let i = 0;
      const interval = setInterval(() => {
        if (i < words.length) {
          updateLastAiMessage(words[i] + " ");
          i++;
        } else {
          clearInterval(interval);
          setIsTyping(false);
        }
      }, 50);
    }, 500);
  };

  const resetChat = () => {
    if (currentChatId === null && messages.length > 1) { // messages.length > 1 so we don't save empty greetings
      const chatTitle = messages.find(m => m.role === 'user')?.content.substring(0, 20) || "Old Chat";
      const newId = Date.now();
      setChatHistory(prev => [{ id: newId, title: chatTitle, data: messages }, ...prev]);
    }
    
    // Reset with greeting
    setMessages([{ role: 'ai', content: "Hello! A new session has started. How can I help?" }]);
    setCurrentChatId(null); 
    setIsTyping(false);
  };

  const handleSwitchChat = (selectedChat) => {
    // 1. If we are already on this chat, do nothing
    if (currentChatId === selectedChat.id) return;

    // 2. If we were on a NEW unsaved chat, save it first
    if (currentChatId === null && messages.length > 0) {
      const chatTitle = messages.find(m => m.role === 'user')?.content.substring(0, 20) || "Old Chat";
      const newId = Date.now();
      setChatHistory(prev => [{ id: newId, title: chatTitle, data: messages }, ...prev]);
    } 
    
    // 3. If we were on an EXISTING chat, update its data in history (in case new messages were added)
    else if (currentChatId !== null) {
      setChatHistory(prev => prev.map(chat => 
        chat.id === currentChatId ? { ...chat, data: messages } : chat
      ));
    }

    // 4. Finally, switch to the selected chat
    setMessages(selectedChat.data);
    setCurrentChatId(selectedChat.id);
    setIsTyping(false);
  };
  
  return (
  <div className="app-container">
    {/* Background Shapes */}
    <div className="bg-decoration">
      <div className="shape circle s1"></div>
      <div className="shape circle s2"></div>
      <div className="shape bug b1"></div>
      <div className="shape bug b2"></div>
    </div>

    <aside className="sidebar">
      <button className="new-chat-btn" onClick={resetChat}>+ New Chat \⁠(⁠๑⁠╹⁠◡⁠╹⁠๑⁠)⁠ﾉ</button>
      <div className="history-list">
        <p className="history-label">Recent Chats (⁠・⁠–⁠・⁠;⁠)⁠ゞ</p>
        {chatHistory.map((chat, index) => (
          <div 
            key={chat.id} 
            className={`history-item ${currentChatId === chat.id ? 'active' : ''}`}
            onClick={() => handleSwitchChat(chat)}
          >
            {chat.title}...
          </div>
        ))}
      </div>
    </aside>

    <main className="chat-main">
      <header className="chat-header">
        <span>Customer Service ヾ⁠(⁠･⁠ω⁠･⁠*⁠)⁠ﾉ</span>
      </header>
      
     <div className="message-list">
      {messages.map((msg, index) => (
        <div key={index} className={`message-row ${msg.role}`}>
          <div className="avatar">
            {msg.role === 'user' ? '(⁠╹⁠▽⁠╹⁠⁠)' : '(⁠≧⁠▽⁠≦⁠)'} 
          </div>
          <div className="message-content">{msg.content}</div>
        </div>
      ))}
      
      {isTyping && (
        <div className="typing-indicator">
          <div className="dot"></div>
          <div className="dot"></div>
          <div className="dot"></div>
          <span>us ko sochne do... (⁠･⁠o⁠･⁠;⁠)</span>
        </div>
      )}
      <div ref={scrollRef} />
    </div>

      <div className="input-area">
        <div className="input-container">
          <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type a message..." 
          />
          <button onClick={handleSend}>Send</button>
        </div>
      </div>
    </main>
  </div>
);
}

export default App;