export type FileSystemItem = {
  id: string;
  type: 'folder' | 'file';
  name:string;
  fileType?: 'pdf' | 'docx' | 'xlsx' | 'png' | 'js';
  lastModified: string;
  mime_type?: string;
  size?: string;
  ingestion_status?: 'pending' | 'processing' | 'completed' | 'failed';
  children?: FileSystemItem[]; // This allows for nesting folders
};

export interface FolderResponse {
  items: FileSystemItem[];
  path: { id: string; name: string }[];
}

export interface SearchFilters {
  query?: string;
  file_type?: string;
  mime_type?: string;
  item_type?: 'file' | 'folder';
  ingestion_status?: 'pending' | 'processing' | 'completed' | 'failed';
  date_from?: string;
  date_to?: string;
  min_size?: number;
  max_size?: number;
}