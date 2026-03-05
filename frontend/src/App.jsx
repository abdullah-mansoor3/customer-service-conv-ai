import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  // const [messages, setMessages] = useState([]); // Stores the chat history
  const [messages, setMessages] = useState([
    { role: 'ai', content: "Hello! I'm your ISP Tech Support Agent. I'm sorry to hear you're having internet issues. Let's get this fixed! Are you currently connected via Wi-Fi or a wired ethernet cable?" }
  ]);
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const socket = useRef(null);
  const scrollRef = useRef(null);
  const IS_OFFLINE = false;

  // 1. Setup WebSocket Connection
  useEffect(() => {
    if (IS_OFFLINE) return; // Don't try to connect if we are testing locally

    // backend/main.py route is /ws/chat (session_id is sent in the JSON payload)
    socket.current = new WebSocket('ws://localhost:8000/ws/chat');

    socket.current.onopen = () => console.log("✅ Connected to AI Backend");

    socket.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Standard streaming pattern: backend sends { "type": "token", "token": "Hello", "done": false }
        if (data.type === 'token') {
          if (data.token) {
            updateLastAiMessage(data.token);
          }
          if (data.done === true) {
            setIsTyping(false);
          }
        }

        // Keep fallback support if the backend was sending "end"
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
        <button 
          className="new-chat-btn" 
          onClick={resetChat}
          disabled={isTyping} // Disable when AI is busy
        >
          + New Chat \⁠(⁠๑⁠╹⁠◡⁠╹⁠๑⁠)⁠ﾉ
        </button>
        <div className="history-list">
          {chatHistory.map((chat) => (
            <div 
              key={chat.id} 
              className={`history-item ${currentChatId === chat.id ? 'active' : ''} ${isTyping ? 'disabled' : ''}`}
              onClick={() => !isTyping && handleSwitchChat(chat)} // Only click if not typing
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
            <div key={index} className={`message-row ${msg.role === 'user' ? 'user-side' : 'ai-side'}`}>
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
            <textarea 
              value={input} 
              onChange={(e) => setInput(e.target.value)}
              onInput={(e) => {
                // This resets height to 'auto' to shrink when deleting, 
                // then sets it to scrollHeight to grow
                e.target.style.height = 'auto';
                e.target.style.height = e.target.scrollHeight + 'px';
              }}
              onKeyPress={(e) => {
                // Only send on Enter (without Shift for new line)
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                  e.target.style.height = 'auto';
                }
              }}
              placeholder={isTyping ? "AI is thinking..." : "Type a message..."}
              disabled={isTyping}
              rows="1" // Start small
            />
            <button 
              onClick={handleSend} 
              disabled={isTyping} // Disable button while typing
            >
              {isTyping ? "..." : "Send"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;