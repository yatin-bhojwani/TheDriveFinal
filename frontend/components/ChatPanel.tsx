'use client';

import { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send } from 'lucide-react';
import { api } from '@/lib/api';
import type { FileSystemItem } from '@/types';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai';
  sources?: { source: string; page?: number }[];
}

interface ChatPanelProps {
    selectedItem: FileSystemItem | null;
    currentFolderId: string;
}

export function ChatPanel({ selectedItem, currentFolderId }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! Select a scope and ask me anything about it.",
      sender: 'ai',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Helper function to get chat context information
  const getChatContext = () => {
    if (selectedItem) {
      if (selectedItem.type === 'file') {
        return {
          placeholder: `Ask about ${selectedItem.name}...`,
          contextName: selectedItem.name,
          icon: '📄'
        };
      } else if (selectedItem.type === 'folder') {
        return {
          placeholder: `Ask about folder "${selectedItem.name}"...`,
          contextName: selectedItem.name,
          icon: '📁'
        };
      }
    }
    
    if (currentFolderId === 'root') {
      return {
        placeholder: "Ask about your entire drive...",
        contextName: "My Drive",
        icon: '💾'
      };
    }
    
    return {
      placeholder: "Ask about this folder...",
      contextName: "Current Folder",
      icon: '📁'
    };
  };

  const { placeholder, contextName, icon } = getChatContext();

  useEffect(() => {
    chatContainerRef.current?.scrollTo({ top: chatContainerRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (input.trim() === '' || isLoading) return;

    const userMessage: Message = { id: Date.now(), text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    
    const query = input;
    setInput('');
    setIsLoading(true);
    
    const aiMessageId = Date.now() + 1;
    setMessages((prev) => [...prev, { id: aiMessageId, text: '', sender: 'ai' }]);

    if (eventSourceRef.current) {
        eventSourceRef.current.close();
    }
    let context: { fileId?: string | null; folderId?: string | null } = {
        folderId: currentFolderId,
    };
    if (selectedItem) {
        if (selectedItem.type === 'file') {
            context = { fileId: selectedItem.id, folderId: null };
        } else { // It's a folder
            context = { fileId: null, folderId: selectedItem.id };
        }
    }
    
    const eventSource = api.getChatStream(query, sessionId, context);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener('token', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.t) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiMessageId ? { ...msg, text: msg.text + data.t } : msg
            )
          );
        }
      } catch (e) {
        console.error("Failed to parse token event:", event.data);
      }
    });

    eventSource.addEventListener('meta', (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.citations && data.citations.length > 0) {
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === aiMessageId ? { ...msg, sources: data.citations } : msg
                    )
                );
            }
        } catch (e) {
            console.error("Failed to parse meta event:", event.data);
        }
    });

    const closeConnection = () => {
      setIsLoading(false);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };

    eventSource.addEventListener('done', closeConnection);
    
    eventSource.onerror = () => {
        setMessages((prev) =>
            prev.map((msg) =>
                msg.id === aiMessageId
                ? { ...msg, text: 'An error occurred with the AI service. Please try again.' }
                : msg
            )
        );
        closeConnection();
    };
  };
  
  const handleTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
  };

  return (
    <aside className="w-96 bg-white dark:bg-gray-800/70 border-l border-gray-200 dark:border-gray-700 flex flex-col shadow-lg">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center gap-3">
        <MessageSquare className="h-6 w-6 text-blue-500" />
        <h3 className="text-lg font-semibold">Chat with Drive</h3>
      </div>

      <div ref={chatContainerRef} className="flex-1 p-4 space-y-6 overflow-y-auto">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex items-start gap-3 ${msg.sender === 'user' ? 'justify-end' : ''}`}>
            {msg.sender === 'ai' && (
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">AI</div>
            )}
            <div className={`p-3 rounded-lg max-w-xs ${
                msg.sender === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-gray-100 dark:bg-gray-700 rounded-tl-none'
            }`}>
              <p className="text-sm whitespace-pre-wrap break-words">{msg.text || (isLoading && msg.id === messages[messages.length - 1].id ? '...' : '')}</p>
              {msg.sender === 'ai' && msg.sources && msg.sources.length > 0 && (
                <div className="text-xs pt-2 mt-2 border-t border-gray-200 dark:border-gray-600">
                  <h4 className="font-bold mb-1">Sources:</h4>
                  <ul className="list-disc pl-4 space-y-1">
                    {msg.sources.map((source, index) => (
                      <li key={index} className="truncate" title={source.source}>
                        {source.source} {source.page && `(p. ${source.page})`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
             {msg.sender === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center font-bold text-sm flex-shrink-0">A</div>
            )}
          </div>
        ))}
        {isLoading && messages[messages.length - 1]?.text === '' && (
             <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">AI</div>
                <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg rounded-tl-none">
                    <div className="animate-pulse flex space-x-1">
                        <div className="h-2 w-2 bg-gray-400 rounded-full"></div>
                        <div className="h-2 w-2 bg-gray-400 rounded-full animation-delay-200"></div>
                        <div className="h-2 w-2 bg-gray-400 rounded-full animation-delay-400"></div>
                    </div>
                </div>
            </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
         <div className="relative">
          <textarea
            placeholder={placeholder}
            className="w-full pl-4 pr-12 py-2 rounded-lg bg-gray-100 dark:bg-gray-700 border border-transparent focus:bg-white dark:focus:bg-gray-800 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all resize-none"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleTextareaKeyDown}
            onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
            }}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:bg-blue-400 disabled:cursor-not-allowed">
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}; 