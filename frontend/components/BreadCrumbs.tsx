'use client';

import { ChevronRight } from 'lucide-react';

interface Breadcrumb {
  id: string;
  name: string;
}

interface BreadcrumbsProps {
  path: Breadcrumb[];
  onNavigate: (id: string) => void;
}

export const Breadcrumbs = ({ path, onNavigate }: BreadcrumbsProps) => {
  return (
    <nav className="flex items-center text-gray-500 dark:text-gray-400 text-sm">
      {path.map((crumb, index) => (
        <div key={crumb.id} className="flex items-center">
          <button
            onClick={() => onNavigate(crumb.id)}
            className="hover:text-blue-500 hover:underline"
            disabled={index === path.length - 1}
          >
            {crumb.name}
          </button>
          {index < path.length - 1 && (
            <ChevronRight className="h-4 w-4 mx-1" />
          )}
        </div>
      ))}
    </nav>
  );
};