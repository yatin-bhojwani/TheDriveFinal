export type FileSystemItem = {
  id: string;
  type: 'folder' | 'file';
  name:string;
  fileType?: 'pdf' | 'docx' | 'xlsx' | 'png' | 'js';
  lastModified: string;
  mime_type?: string;
  size?: string;
  children?: FileSystemItem[]; // This allows for nesting folders
};

export interface FolderResponse {
  items: FileSystemItem[];
  path: { id: string; name: string }[];
}