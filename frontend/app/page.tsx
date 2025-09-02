'use client';

import { Suspense, useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { LayoutGrid, List } from 'lucide-react';
import { FileGridItem } from '@/components/FileGridItem';
import { FileListItem } from '@/components/FileListItem';
import { ChatPanel } from '@/components/ChatPanel';
import { Breadcrumbs } from '@/components/BreadCrumbs';
import { Header } from '@/components/Header';
import { CreateFolderModal } from '@/components/CreateFolderModal';
import { RenameModal } from '@/components/RenameModal';
import { ContextMenu } from '@/components/ContextMenu';
import type { FileSystemItem } from '@/types';
import { api } from '@/lib/api';
import { FileViewer } from '@/components/FileViewer';
import { useAuth } from '@/hooks/useAuth';

function DriveUI() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [items, setItems] = useState<FileSystemItem[]>([]);
  const [path, setPath] = useState<{id: string, name: string}[]>([{ id: 'root', name: 'My Drive' }]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewingFile, setViewingFile] = useState<{ name: string; mime_type?: string; url: string | null } | null>(null);
  const [view, setView] = useState<'grid' | 'list'>('grid');
  const [isChatOpen, setChatOpen] = useState(false);
  const [isCreateFolderModalOpen, setCreateFolderModalOpen] = useState(false);
  const [itemToRename, setItemToRename] = useState<FileSystemItem | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number, y: number, item: FileSystemItem } | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [selectedItem, setSelectedItem] = useState<FileSystemItem | null>(null);
  const currentFolderId = searchParams.get('path') || 'root';

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const res = await api.getItems(currentFolderId);
        setItems(res.items);
        setPath(res.path);
      } catch (err) {
        console.error(err);
        if (err instanceof Error && err.message.includes("not found")) {
          // Current folder doesn't exist, redirect to root
          router.push('/');
        }
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [currentFolderId, router]);

  // A single click selects an item
  const handleItemClick = (item: FileSystemItem) => {
    setSelectedItem(item);
  };

  // A double click opens a folder or views a file
  const handleItemDoubleClick = async (item: FileSystemItem) => {
    if (item.type === 'folder') {
      router.push(`/?path=${item.id}`);
      setSelectedItem(null); // Deselect when navigating into a folder
    } else {
      setViewingFile({ name: item.name, mime_type: item.mime_type, url: null });
      try {
        const { url } = await api.getViewLink(item.id);
        setViewingFile({ name: item.name, mime_type: item.mime_type, url });
      } catch (error) {
        console.error("Could not get view link:", error);
        setViewingFile(null);
      }
    }
  };

  const handleBreadcrumbNavigate = (id: string) => {
      router.push(`/?path=${id}`);
      setSelectedItem(null); // Deselect when navigating
  };

  const handleUpload = async (file: File) => {
     try {
        setUploadStatus("uploading");
        const newItem = await api.uploadFile(file, currentFolderId);
        setItems(prev => [...prev, newItem]);
        setUploadStatus("success");
    } catch (error) {
        console.error("Upload failed:", error);
        if (error instanceof Error && error.message.includes("Parent folder not found")) {
            // Parent folder was deleted, redirect to root
            router.push('/');
        }
        setUploadStatus("error");
    } finally {
        setTimeout(() => setUploadStatus("idle"), 3000);
    }
  };

  const handleConfirmCreateFolder = async (folderName: string) => {
    try {
        const newFolder = await api.createFolder(folderName, currentFolderId);
        setItems(prev => [...prev, newFolder]);
        setCreateFolderModalOpen(false);
    } catch (error) {
        console.error("Failed to create folder:", error);
        if (error instanceof Error && error.message.includes("Parent folder not found")) {
            // Parent folder was deleted, redirect to root
            router.push('/');
        }
    }
  };

  const handleRenameItem = async (newName: string) => {
    if (!itemToRename) return;
    try {
        const updatedItem = await api.renameItem(itemToRename.id, newName);
        setItems(prev => prev.map(item => item.id === updatedItem.id ? { ...item, ...updatedItem } : item));
        setItemToRename(null);
    } catch (error) {
        console.error("Rename failed:", error);
    }
  };

  const handleDeleteItem = async () => {
    if (!contextMenu) return;
    if (confirm(`Are you sure you want to delete "${contextMenu.item.name}"?`)) {
        try {
            await api.deleteItem(contextMenu.item.id);
            setItems(prev => prev.filter(item => item.id !== contextMenu.item.id));
        } catch (error) {
            console.error("Delete failed:", error);
        }
    }
    setContextMenu(null);
  };
  
  const isNameTaken = (name: string, idToExclude: string = ''): boolean => {
    return items.some(item => item.id !== idToExclude && item.name.toLowerCase() === name.toLowerCase());
  };
  
  const handleContextMenu = (event: React.MouseEvent, item: FileSystemItem) => {
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY, item });
  };

  if (isLoading) {
    return <div className="h-screen w-full flex items-center justify-center dark:bg-gray-900 dark:text-white">Loading Drive...</div>
  }

  return (
      <div className="bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 h-screen flex flex-col font-sans" onClick={() => setContextMenu(null)}>
      <Header 
        onUpload={handleUpload} 
        onCreateFolder={() => setCreateFolderModalOpen(true)} 
        onToggleChat={() => setChatOpen(!isChatOpen)} 
      />
      
      {uploadStatus !== "idle" && (
        <div className={`fixed bottom-5 right-5 px-4 py-2 rounded shadow-lg z-50
        ${uploadStatus === "uploading" ? "bg-yellow-500 text-white" : ""}
        ${uploadStatus === "success" ? "bg-green-500 text-white" : ""}
        ${uploadStatus === "error" ? "bg-red-500 text-white" : ""}
    `   }>
        {uploadStatus === "uploading" && "Uploading & Processing... This may take a few minutes for large files."}
        {uploadStatus === "success" && "File uploaded and processed successfully!"}
        {uploadStatus === "error" && "Upload failed! Please try again."}
      </div>
      )}
      
      <div className="flex flex-1 overflow-hidden">
        <main className="flex-1 p-6 overflow-y-auto transition-all duration-300">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-2xl font-semibold">{path[path.length - 1]?.name || 'My Drive'}</h2>
              <Breadcrumbs path={path} onNavigate={handleBreadcrumbNavigate} /> 
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setView('grid')} className={`p-2 rounded-md ${view === 'grid' ? 'bg-gray-200 dark:bg-gray-700' : 'hover:bg-gray-200 dark:hover:bg-gray-700'}`}><LayoutGrid className="h-5 w-5" /></button>
              <button onClick={() => setView('list')} className={`p-2 rounded-md ${view === 'list' ? 'bg-gray-200 dark:bg-gray-700' : 'hover:bg-gray-200 dark:hover:bg-gray-700'}`}><List className="h-5 w-5" /></button>
            </div>
          </div>
          
          {view === 'grid' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
              {items.map((item) => <FileGridItem 
                key={item.id} 
                item={item} 
                isSelected={selectedItem?.id === item.id}
                onItemClick={handleItemClick}
                onItemDoubleClick={handleItemDoubleClick} 
                onContextMenu={handleContextMenu} />)}
            </div>
          ) : (
            <div>
                <div className="flex items-center w-full p-2 text-sm font-semibold text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
                    <div className="w-1/2">Name</div>
                    <div className="w-1/4">Last Modified</div>
                    <div className="w-1/4">File Size</div>
                </div>
                <div className="flex flex-col">
                    {items.map((item) => <FileListItem 
                        key={item.id} 
                        item={item} 
                        isSelected={selectedItem?.id === item.id}
                        onItemClick={handleItemClick} 
                        onItemDoubleClick={handleItemDoubleClick} 
                        onContextMenu={handleContextMenu} />)}
                </div>
            </div>
          )}
        </main>

        {isChatOpen && <ChatPanel selectedItem={selectedItem} currentFolderId={currentFolderId} />}
      </div>
      
      {viewingFile && <FileViewer file={viewingFile} onClose={() => setViewingFile(null)} />}
      {contextMenu && <ContextMenu x={contextMenu.x} y={contextMenu.y} onRename={() => { setItemToRename(contextMenu.item); setContextMenu(null); }} onDelete={handleDeleteItem} />}
      <CreateFolderModal isOpen={isCreateFolderModalOpen} onClose={() => setCreateFolderModalOpen(false)} onCreate={handleConfirmCreateFolder} isNameTaken={(name) => isNameTaken(name)} />
      <RenameModal item={itemToRename} onClose={() => setItemToRename(null)} onRename={handleRenameItem} isNameTaken={(name, id) => isNameTaken(name, id)} />
    </div>
  );
}

export default function DrivePage() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading || !isAuthenticated) {
    return <div className="h-screen w-full flex items-center justify-center dark:bg-gray-900 dark:text-white">Authenticating...</div>;
  }
  
  return (
    <Suspense fallback={<div></div>}>
      <DriveUI />
    </Suspense>
  );
}