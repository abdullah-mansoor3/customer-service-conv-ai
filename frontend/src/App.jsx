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

  // Voice-related state
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [audioChunks, setAudioChunks] = useState([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);

  const socket = useRef(null);
  const voiceSocket = useRef(null);
  const mediaRecorder = useRef(null);
  const audioContext = useRef(null);
  const scrollRef = useRef(null);
  const IS_OFFLINE = false;

  const getWsBaseUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // If served via nginx at :3000, use same host so /ws proxy works.
    if (window.location.port === '3000') {
      return `${protocol}//${window.location.host}`;
    }
    // Vite dev server (:5173) should still hit backend directly on :8000.
    return `${protocol}//${window.location.hostname}:8000`;
  };

  // 1. Setup WebSocket Connection
  useEffect(() => {
    if (IS_OFFLINE) return; // Don't try to connect if we are testing locally

    // backend/main.py route is /ws/chat (session_id is sent in the JSON payload)
    socket.current = new WebSocket(`${getWsBaseUrl()}/ws/chat`);

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

  // 2. Setup Voice WebSocket Connection
  useEffect(() => {
    if (IS_OFFLINE || !isVoiceMode) return;

    voiceSocket.current = new WebSocket(`${getWsBaseUrl()}/ws/voice-chat`);

    voiceSocket.current.onopen = () => console.log("✅ Connected to Voice Backend");

    voiceSocket.current.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // JSON message (transcription or error)
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'transcription') {
            // Show what the user said
            setMessages(prev => [...prev, { role: 'user', content: data.text }]);
            setIsTyping(true);
          } else if (data.type === 'assistant_text') {
            setMessages(prev => [...prev, { role: 'ai', content: data.text }]);
            setIsTyping(false);
          } else if (data.type === 'error') {
            console.error("Voice error:", data.error);
            setMessages(prev => [...prev, { role: 'ai', content: `Voice Error: ${data.error}` }]);
            setIsTyping(false);
          }
        } catch (err) {
          console.error("Voice JSON parse error:", err);
        }
      } else {
        // Binary audio data - play it
        setIsTyping(false);
        playAudioResponse(event.data);
      }
    };

    voiceSocket.current.onclose = () => {
      console.log("❌ Voice WebSocket disconnected");
      setIsTyping(false);
    };

    return () => voiceSocket.current?.close();
  }, [IS_OFFLINE, isVoiceMode]);

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

  // Voice functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      const chunks = [];
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        sendAudioToBackend(audioBlob);
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      setMessages(prev => [...prev, { role: 'ai', content: "❌ Error: Could not access microphone." }]);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    if (!voiceSocket.current || voiceSocket.current.readyState !== WebSocket.OPEN) {
      setMessages(prev => [...prev, { role: 'ai', content: "❌ Error: Voice connection not available." }]);
      return;
    }

    try {
      // Convert blob to array buffer
      const arrayBuffer = await audioBlob.arrayBuffer();
      voiceSocket.current.send(arrayBuffer);
    } catch (error) {
      console.error('Error sending audio:', error);
      setMessages(prev => [...prev, { role: 'ai', content: "❌ Error: Failed to send audio." }]);
    }
  };

  const playAudioResponse = async (audioData) => {
    try {
      setIsPlayingAudio(true);

      // Create blob from received data
      const audioBlob = new Blob([audioData], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);

      const audio = new Audio(audioUrl);
      audio.onended = () => {
        setIsPlayingAudio(false);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
    } catch (error) {
      console.error('Error playing audio:', error);
      setIsPlayingAudio(false);
    }
  };

  const toggleVoiceMode = () => {
    setIsVoiceMode(!isVoiceMode);
    if (isVoiceMode) {
      // Switching to text mode
      stopRecording();
    }
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
            {!isVoiceMode ? (
              <>
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
              </>
            ) : (
              <div className="voice-controls">
                <button
                  className={`voice-btn ${isRecording ? 'recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isTyping || isPlayingAudio}
                >
                  {isRecording ? '⏹️ Stop' : '🎤 Speak'}
                </button>
                <span className="voice-status">
                  {isRecording ? 'Listening...' : isPlayingAudio ? 'Speaking...' : 'Voice Mode'}
                </span>
              </div>
            )}
            <button
              className={`mode-toggle ${isVoiceMode ? 'voice-active' : ''}`}
              onClick={toggleVoiceMode}
              disabled={isTyping}
              title={isVoiceMode ? 'Switch to Text Mode' : 'Switch to Voice Mode'}
            >
              {isVoiceMode ? '📝' : '🎤'}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;