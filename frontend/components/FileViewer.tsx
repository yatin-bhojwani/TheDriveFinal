'use client';

import { X, Loader2, Wand2, ClipboardPaste } from 'lucide-react';
import { useState, useEffect } from 'react';

// --- New "Paste" Modal ---
// This modal opens immediately when the floating button is clicked.
const AiPasteModal = ({ onClose }: { onClose: () => void }) => {
  const [pastedText, setPastedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiResponse, setAiResponse] = useState('');

  const handleAskAi = () => {
    if (!pastedText) return;
    setIsLoading(true);
    setAiResponse('');
    setTimeout(() => {
      const dummyResponse = `Based on the pasted text, the key insight is that explicit user actions, such as pasting into a text field, lead to more reliable and understandable user experiences than implicit event listeners. This method ensures functionality across all platforms without depending on complex browser APIs.`;
      setAiResponse(dummyResponse);
      setIsLoading(false);
    }, 2000);
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black bg-opacity-50"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-lg bg-white p-6 shadow-2xl dark:bg-gray-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Ask AI</h3>
          <button onClick={onClose} className="rounded-full p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Show the AI Response if it exists */}
        {aiResponse ? (
          <>
            <p className="mt-4 text-sm font-medium text-gray-800 dark:text-gray-200">AI Response:</p>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">{aiResponse}</p>
          </>
        ) : (
          // Otherwise, show the text area for pasting
          <>
            <p className="mt-4 text-sm text-gray-800 dark:text-gray-200">
              Paste the text you copied from the PDF below.
            </p>
            <textarea
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder="Right-click and paste, or press Ctrl+V..."
              className="mt-2 h-32 w-full rounded-md border border-gray-300 p-3 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
            />
          </>
        )}

        <div className="mt-6 flex justify-end">
          <button
            onClick={handleAskAi}
            disabled={isLoading || !pastedText}
            className="flex min-w-[140px] items-center justify-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-gray-400"
          >
            {isLoading ? (
              <><Loader2 className="h-4 w-4 animate-spin" /><span>Thinking...</span></>
            ) : (
              <><Wand2 className="h-4 w-4" /><span>{aiResponse ? 'Ask Again' : 'Submit to AI'}</span></>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Main FileViewer Component ---

interface FileViewerProps {
  file: { name: string; mime_type?: string; url: string | null };
  onClose: () => void;
}

export const FileViewer = ({ file, onClose }: FileViewerProps) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const isPdf = file.mime_type === 'application/pdf' || /\.pdf$/i.test(file.name);
  const isImage = file.mime_type?.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp|bmp|svg)$/i.test(file.name);
  const isText = file.mime_type?.startsWith('text/') || /\.(txt|md|json|csv|xml|yaml|yml|log)$/i.test(file.name);
  const isCode = /\.(js|jsx|ts|tsx|py|java|cpp|c|html|css|scss|php|rb|go|rs|swift|kt)$/i.test(file.name);
  const isVideo = file.mime_type?.startsWith('video/') || /\.(mp4|webm|ogg|mov|avi|mkv)$/i.test(file.name);
  const isAudio = file.mime_type?.startsWith('audio/') || /\.(mp3|wav|ogg|flac|aac|m4a)$/i.test(file.name);
  const isOffice = /\.(docx?|xlsx?|pptx?)$/i.test(file.name) || 
                   file.mime_type?.includes('officedocument') ||
                   file.mime_type?.includes('msword') ||
                   file.mime_type?.includes('spreadsheet') ||
                   file.mime_type?.includes('presentation');

  const renderFileContent = () => {
    if (!file.url) {
      return <Loader2 className="h-8 w-8 animate-spin" />;
    }

    if (isPdf) {
      return (
        <div className="relative h-full w-full">
          <iframe src={file.url} width="100%" height="100%" className="bg-gray-100" title={file.name} />
        </div>
      );
    }

    if (isImage) {
      return <img src={file.url} alt={file.name} className="max-w-full max-h-full object-contain" />;
    }

    if (isVideo) {
      return (
        <video controls className="max-w-full max-h-full">
          <source src={file.url} type={file.mime_type || 'video/mp4'} />
          Your browser does not support the video tag.
        </video>
      );
    }

    if (isAudio) {
      return (
        <div className="text-center p-8">
          <div className="mb-4">
            <div className="text-6xl mb-4">🎵</div>
            <h3 className="text-lg font-semibold">{file.name}</h3>
          </div>
          <audio controls className="w-full max-w-md">
            <source src={file.url} type={file.mime_type || 'audio/mp3'} />
            Your browser does not support the audio tag.
          </audio>
        </div>
      );
    }

    if (isText || isCode) {
      return (
        <div className="h-full w-full p-4 overflow-auto">
          <iframe 
            src={file.url} 
            width="100%" 
            height="100%" 
            className="bg-white border-0" 
            title={file.name}
            style={{ minHeight: '600px' }}
          />
        </div>
      );
    }

    if (isOffice) {
      // For Office documents, try to use Office Online Viewer
      const officeViewerUrl = `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(file.url)}`;
      return (
        <div className="h-full w-full">
          <iframe 
            src={officeViewerUrl} 
            width="100%" 
            height="100%" 
            className="bg-gray-100" 
            title={file.name}
          />
        </div>
      );
    }

    // Fallback for unsupported file types
    return (
      <div className="text-center p-8">
        <div className="text-6xl mb-4">📄</div>
        <h3 className="text-lg font-semibold mb-2">{file.name}</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Preview not available for this file type.
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-500 mb-4">
          {file.mime_type || 'Unknown file type'}
        </p>
        <a 
          href={file.url} 
          download={file.name} 
          className="inline-block rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
        >
          Download File
        </a>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black bg-opacity-75 p-4" onClick={onClose}>
      <header className="flex items-center justify-between text-white mb-4">
        <h2 className="text-lg font-semibold truncate pr-4">{file.name}</h2>
        <button onClick={onClose} className="rounded-full p-2 hover:bg-white/20">
          <X className="h-6 w-6" />
        </button>
      </header>
      <div className="flex flex-1 items-center justify-center overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex h-full max-h-[90vh] w-full max-w-6xl items-center justify-center rounded-lg bg-white dark:bg-gray-800">
          {renderFileContent()}
        </div>
      </div>

      {/* Renders the Floating Action Button only for PDFs */}
      {isPdf && (
        <button
          onClick={() => setIsModalOpen(true)}
          title="Ask AI about text from the PDF"
          className="group fixed bottom-6 right-6 z-[60] flex h-14 w-14 items-center justify-center rounded-full bg-blue-600 text-white shadow-lg transition-transform hover:scale-110 hover:bg-blue-500"
        >
          <Wand2 className="h-7 w-7" />
        </button>
      )}

      {/* Renders the new "Paste" modal when the button is clicked */}
      {isModalOpen && (
        <AiPasteModal onClose={() => setIsModalOpen(false)} />
      )}
    </div>
  );
};