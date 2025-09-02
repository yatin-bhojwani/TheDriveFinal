'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export const useAuth = (redirectTo = '/login') => {
    const router = useRouter();
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const checkAuth = async () => {
            try {
                await api.getMe();
                setIsAuthenticated(true);
            } catch (error) {
                setIsAuthenticated(false);
                if (window.location.pathname !== redirectTo) {
                    router.replace(redirectTo);
                }
            } finally {
                setIsLoading(false);
            }
        };

        checkAuth();
    }, [router, redirectTo]);

    return { isAuthenticated, isLoading };
};