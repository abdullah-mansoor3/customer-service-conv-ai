import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const createSessionId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

function App() {
  // const [messages, setMessages] = useState([]); // Stores the chat history
  const [messages, setMessages] = useState([
    { role: 'ai', content: "Hello! I'm your ISP Tech Support Agent. I'm sorry to hear you're having internet issues. Let's get this fixed! Are you currently connected via Wi-Fi or a wired ethernet cable?" }
  ]);
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [activeUiChatId, setActiveUiChatId] = useState(null);
  const [backendSessionId, setBackendSessionId] = useState(() => createSessionId());
  const [isTyping, setIsTyping] = useState(false);

  // Voice-related state
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isVoiceProcessing, setIsVoiceProcessing] = useState(false);
  const [isVoiceTurnActive, setIsVoiceTurnActive] = useState(false);
  const [isVoiceLoading, setIsVoiceLoading] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [waveLevels, setWaveLevels] = useState(Array(24).fill(0.08));
  const [voiceOptions, setVoiceOptions] = useState([
    { id: 'ella', name: 'Ella', description: 'US English female, warm and clear' },
    { id: 'john', name: 'John', description: 'US English male, neutral and calm' },
  ]);
  const [selectedVoiceId, setSelectedVoiceId] = useState('ella');

  const socket = useRef(null);
  const voiceSocket = useRef(null);
  const mediaRecorder = useRef(null);
  const audioContext = useRef(null);
  const analyserRef = useRef(null);
  const waveformFrameRef = useRef(null);
  const micStreamRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const discardRecordingRef = useRef(false);
  const currentAudioRef = useRef(null);
  const transcriptTargetWordsRef = useRef([]);
  const transcriptRenderedCountRef = useRef(0);
  const transcriptTickerRef = useRef(null);
  const transcriptFinalizePendingRef = useRef(false);
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

        if (data.type === 'session_id' && data.session_id) {
          // Keep frontend and backend aligned, but frontend still owns session lifecycle.
          setBackendSessionId(data.session_id);
        }

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
          if (data.type === 'transcription_partial' || data.type === 'transcription_final') {
            handleTranscriptProgress(data.text || '');
            if (data.type === 'transcription_final') {
              transcriptFinalizePendingRef.current = true;
              setIsVoiceProcessing(true);
              setIsTyping(true);
            }
          } else if (data.type === 'voice_catalog') {
            if (Array.isArray(data.voices) && data.voices.length > 0) {
              setVoiceOptions(data.voices);
            }
            if (data.default_voice_id) {
              setSelectedVoiceId(data.default_voice_id);
            }
          } else if (data.type === 'voice_selected') {
            if (data.voice_id) {
              setSelectedVoiceId(data.voice_id);
            }
          } else if (data.type === 'session_id') {
            if (data.session_id) {
              setBackendSessionId(data.session_id);
            }
          } else if (data.type === 'voice_loading') {
            setIsVoiceLoading(true);
          } else if (data.type === 'voice_ready') {
            setIsVoiceLoading(false);
          } else if (data.type === 'assistant_text') {
            setMessages(prev => [...prev, { role: 'ai', content: data.text }]);
            setIsVoiceProcessing(false);
            setIsTyping(false);
          } else if (data.type === 'cancelled') {
            resetLiveVoiceState();
            setIsVoiceTurnActive(false);
            setIsVoiceProcessing(false);
            setMessages(prev => [...prev, { role: 'ai', content: 'Voice turn cancelled.' }]);
          } else if (data.type === 'error') {
            console.error("Voice error:", data.error);
            setIsVoiceTurnActive(false);
            setIsVoiceProcessing(false);
            setMessages(prev => [...prev, { role: 'ai', content: `Voice Error: ${data.error}` }]);
            setIsTyping(false);
          }
        } catch (err) {
          console.error("Voice JSON parse error:", err);
        }
      } else {
        // Binary audio data - play it
        setIsTyping(false);
        setIsVoiceProcessing(false);
        playAudioResponse(event.data);
      }
    };

    voiceSocket.current.onclose = () => {
      console.log("❌ Voice WebSocket disconnected");
      setIsVoiceTurnActive(false);
      setIsVoiceProcessing(false);
      setIsVoiceLoading(false);
      stopWaveAnimation(true);
      setIsTyping(false);
    };

    return () => voiceSocket.current?.close();
  }, [IS_OFFLINE, isVoiceMode]);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (transcriptTickerRef.current) {
        clearInterval(transcriptTickerRef.current);
      }
      stopWaveAnimation(true);
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
      }
    };
  }, []);

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

  const upsertLiveUserMessage = (text, finalize = false) => {
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.role === 'user' && m.isLiveTranscription);
      if (idx === -1) {
        return [...prev, { role: 'user', content: text, isLiveTranscription: !finalize }];
      }
      const updated = [...prev];
      updated[idx] = {
        ...updated[idx],
        content: text,
        isLiveTranscription: !finalize,
      };
      return updated;
    });
  };

  const resetLiveVoiceState = () => {
    transcriptTargetWordsRef.current = [];
    transcriptRenderedCountRef.current = 0;
    transcriptFinalizePendingRef.current = false;
    setLiveTranscript('');
    if (transcriptTickerRef.current) {
      clearInterval(transcriptTickerRef.current);
      transcriptTickerRef.current = null;
    }
    setMessages((prev) => prev.map((m) => (
      m.isLiveTranscription ? { ...m, isLiveTranscription: false } : m
    )));
  };

  const ensureTranscriptTicker = () => {
    if (transcriptTickerRef.current) return;
    transcriptTickerRef.current = setInterval(() => {
      const target = transcriptTargetWordsRef.current;
      const rendered = transcriptRenderedCountRef.current;

      if (rendered < target.length) {
        const nextCount = rendered + 1;
        transcriptRenderedCountRef.current = nextCount;
        const text = target.slice(0, nextCount).join(' ');
        setLiveTranscript(text);
        upsertLiveUserMessage(text, false);
        return;
      }

      if (transcriptFinalizePendingRef.current && target.length > 0) {
        transcriptFinalizePendingRef.current = false;
        const finalText = target.join(' ');
        setLiveTranscript(finalText);
        upsertLiveUserMessage(finalText, true);
      }

      if (!transcriptFinalizePendingRef.current && rendered >= target.length) {
        clearInterval(transcriptTickerRef.current);
        transcriptTickerRef.current = null;
      }
    }, 70);
  };

  const handleTranscriptProgress = (text) => {
    const words = text.trim().split(/\s+/).filter(Boolean);
    transcriptTargetWordsRef.current = words;
    ensureTranscriptTicker();
  };

  const stopWaveAnimation = (reset = false) => {
    if (waveformFrameRef.current) {
      cancelAnimationFrame(waveformFrameRef.current);
      waveformFrameRef.current = null;
    }
    if (audioContext.current) {
      audioContext.current.close().catch(() => {});
      audioContext.current = null;
    }
    analyserRef.current = null;
    if (reset) {
      setWaveLevels(Array(24).fill(0.08));
    }
  };

  const startWaveAnimation = (stream) => {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;

    audioContext.current = new Ctx();
    const analyser = audioContext.current.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.85;
    const source = audioContext.current.createMediaStreamSource(stream);
    source.connect(analyser);
    analyserRef.current = analyser;

    const bins = new Uint8Array(analyser.frequencyBinCount);
    const draw = () => {
      if (!analyserRef.current) return;
      analyserRef.current.getByteFrequencyData(bins);

      const sampleCount = 24;
      const step = Math.max(1, Math.floor(bins.length / sampleCount));
      const next = Array.from({ length: sampleCount }, (_, i) => {
        const val = bins[i * step] / 255;
        return Math.max(0.08, val);
      });

      setWaveLevels(next);
      waveformFrameRef.current = requestAnimationFrame(draw);
    };

    waveformFrameRef.current = requestAnimationFrame(draw);
  };

  // Voice functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;
      startWaveAnimation(stream);
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      discardRecordingRef.current = false;
      recordingChunksRef.current = [];
      resetLiveVoiceState();

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.current.onstop = () => {
        stopWaveAnimation(true);
        const shouldDiscard = discardRecordingRef.current;
        if (!shouldDiscard) {
          const audioBlob = new Blob(recordingChunksRef.current, { type: 'audio/webm' });
          sendAudioToBackend(audioBlob);
        }
        if (micStreamRef.current) {
          micStreamRef.current.getTracks().forEach((track) => track.stop());
          micStreamRef.current = null;
        }
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
      setIsVoiceProcessing(true);
    }
  };

  const cancelRecording = () => {
    discardRecordingRef.current = true;
    setIsRecording(false);
    setIsVoiceTurnActive(false);
    setIsVoiceProcessing(false);
    setIsTyping(false);
    resetLiveVoiceState();
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    stopWaveAnimation(true);
  };

  const sendAudioToBackend = async (audioBlob) => {
    if (isVoiceTurnActive) {
      setMessages(prev => [...prev, { role: 'ai', content: 'Please wait, previous voice response is still in progress.' }]);
      return;
    }

    if (!voiceSocket.current || voiceSocket.current.readyState !== WebSocket.OPEN) {
      setMessages(prev => [...prev, { role: 'ai', content: "❌ Error: Voice connection not available." }]);
      setIsVoiceProcessing(false);
      setIsVoiceTurnActive(false);
      return;
    }

    try {
      setIsVoiceTurnActive(true);

      // Ensure voice and text share the same backend session/state.
      voiceSocket.current.send(JSON.stringify({
        type: 'set_session',
        session_id: backendSessionId,
      }));

      // Ensure backend uses the currently selected voice for this turn.
      voiceSocket.current.send(JSON.stringify({ type: 'set_voice', voice_id: selectedVoiceId }));

      // Convert blob to array buffer
      const arrayBuffer = await audioBlob.arrayBuffer();
      voiceSocket.current.send(arrayBuffer);
    } catch (error) {
      console.error('Error sending audio:', error);
      setMessages(prev => [...prev, { role: 'ai', content: "❌ Error: Failed to send audio." }]);
      setIsVoiceTurnActive(false);
      setIsVoiceProcessing(false);
    }
  };

  const handleVoiceChange = (nextVoiceId) => {
    setSelectedVoiceId(nextVoiceId);
    setIsVoiceLoading(true);
    if (voiceSocket.current && voiceSocket.current.readyState === WebSocket.OPEN) {
      voiceSocket.current.send(JSON.stringify({ type: 'set_voice', voice_id: nextVoiceId }));
    } else {
      setIsVoiceLoading(false);
    }
  };

  const playAudioResponse = async (audioData) => {
    try {
      setIsPlayingAudio(true);

      // Create blob from received data
      const audioBlob = new Blob([audioData], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);

      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;
      audio.onended = () => {
        setIsPlayingAudio(false);
        setIsVoiceTurnActive(false);
        currentAudioRef.current = null;
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
    } catch (error) {
      console.error('Error playing audio:', error);
      setIsPlayingAudio(false);
      setIsVoiceTurnActive(false);
    }
  };

  const cancelVoiceTurn = () => {
    if (isRecording) {
      cancelRecording();
      return;
    }

    if (isPlayingAudio && currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
      setIsPlayingAudio(false);
      setIsVoiceTurnActive(false);
      return;
    }

    if (isVoiceProcessing && voiceSocket.current && voiceSocket.current.readyState === WebSocket.OPEN) {
      voiceSocket.current.send(JSON.stringify({ type: 'cancel_current' }));
      setIsVoiceTurnActive(false);
      setIsVoiceProcessing(false);
      setIsTyping(false);
      resetLiveVoiceState();
    }
  };

  const toggleVoiceMode = () => {
    setIsVoiceMode(!isVoiceMode);
    if (isVoiceMode) {
      // Switching to text mode
      cancelVoiceTurn();
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
          session_id: backendSessionId
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
    if (activeUiChatId === null && messages.length > 1) { // messages.length > 1 so we don't save empty greetings
      const chatTitle = messages.find(m => m.role === 'user')?.content.substring(0, 20) || "Old Chat";
      const newId = Date.now();
      setChatHistory(prev => [{ id: newId, title: chatTitle, data: messages, sessionId: backendSessionId }, ...prev]);
    }

    // Reset with greeting
    setMessages([{ role: 'ai', content: "Hello! A new session has started. How can I help?" }]);
    setActiveUiChatId(null);
    setBackendSessionId(createSessionId());
    setIsTyping(false);
  };

  const handleSwitchChat = (selectedChat) => {
    // 1. If we are already on this chat, do nothing
    if (activeUiChatId === selectedChat.id) return;

    // 2. If we were on a NEW unsaved chat, save it first
    if (activeUiChatId === null && messages.length > 0) {
      const chatTitle = messages.find(m => m.role === 'user')?.content.substring(0, 20) || "Old Chat";
      const newId = Date.now();
      setChatHistory(prev => [{ id: newId, title: chatTitle, data: messages, sessionId: backendSessionId }, ...prev]);
    }

    // 3. If we were on an EXISTING chat, update its data in history (in case new messages were added)
    else if (activeUiChatId !== null) {
      setChatHistory(prev => prev.map(chat =>
        chat.id === activeUiChatId ? { ...chat, data: messages, sessionId: backendSessionId } : chat
      ));
    }

    // 4. Finally, switch to the selected chat
    setMessages(selectedChat.data);
    setActiveUiChatId(selectedChat.id);
    setBackendSessionId(selectedChat.sessionId || createSessionId());
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
              className={`history-item ${activeUiChatId === chat.id ? 'active' : ''} ${isTyping ? 'disabled' : ''}`}
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
                <div className="voice-picker">
                  <label htmlFor="voice-select">Voice</label>
                  <select
                    id="voice-select"
                    value={selectedVoiceId}
                    onChange={(e) => handleVoiceChange(e.target.value)}
                    disabled={isRecording || isVoiceProcessing || isPlayingAudio || isVoiceTurnActive}
                  >
                    {voiceOptions.map((voice) => (
                      <option key={voice.id} value={voice.id}>
                        {voice.name}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  className={`voice-btn ${isRecording ? 'recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isPlayingAudio || isVoiceProcessing || isVoiceTurnActive}
                >
                  {isRecording ? '⏹️ Stop' : '🎤 Speak'}
                </button>
                {(isRecording || liveTranscript) && (
                  <div className="voice-live-area">
                    <div className="voice-waveform" aria-label="voice waveform">
                      {waveLevels.map((level, i) => (
                        <span
                          key={i}
                          className="voice-wave-bar"
                          style={{ transform: `scaleY(${Math.min(1, Math.max(0.08, level))})` }}
                        />
                      ))}
                    </div>
                    <div className="voice-transcript-live">
                      {liveTranscript || 'Listening...'}
                    </div>
                  </div>
                )}
                <span className="voice-status">
                  {isVoiceLoading
                    ? 'Preparing voice...'
                    : isRecording
                    ? 'Listening...'
                    : isVoiceProcessing
                      ? 'Transcribing...'
                      : isPlayingAudio
                        ? 'Speaking...'
                        : 'Voice Mode'}
                </span>
                {(isRecording || isVoiceProcessing || isPlayingAudio) && (
                  <button className="voice-cancel-btn" onClick={cancelVoiceTurn}>
                    Cancel
                  </button>
                )}
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