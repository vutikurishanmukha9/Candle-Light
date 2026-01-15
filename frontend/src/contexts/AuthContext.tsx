/**
 * Authentication Context
 * 
 * Provides authentication state and methods throughout the app.
 * 
 * SECURITY NOTE: Tokens are currently stored in localStorage for simplicity.
 * For production with high security requirements, consider:
 * - Using httpOnly cookies for refresh tokens (requires backend changes)
 * - Keeping access tokens in memory only
 * - Implementing PKCE for additional security
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authApi, User, isAuthenticated, clearTokens, getAccessToken } from '@/services/api';

interface AuthContextType {
    user: User | null;
    isLoading: boolean;
    isLoggedIn: boolean;
    login: (email: string, password: string) => Promise<void>;
    register: (email: string, password: string, fullName?: string) => Promise<void>;
    logout: () => Promise<void>;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    const refreshUser = useCallback(async () => {
        if (isAuthenticated()) {
            try {
                const userData = await authApi.getCurrentUser();
                setUser(userData);
            } catch (error) {
                console.error('Failed to fetch user:', error);
                clearTokens();
                setUser(null);
            }
        } else {
            setUser(null);
        }
    }, []);

    // Check auth status on mount
    useEffect(() => {
        const initAuth = async () => {
            setIsLoading(true);
            await refreshUser();
            setIsLoading(false);
        };
        initAuth();
    }, [refreshUser]);

    const login = async (email: string, password: string) => {
        await authApi.login({ email, password });
        await refreshUser();
    };

    const register = async (email: string, password: string, fullName?: string) => {
        await authApi.register({ email, password, full_name: fullName });
        // Auto-login after registration
        await authApi.login({ email, password });
        await refreshUser();
    };

    const logout = async () => {
        await authApi.logout();
        setUser(null);
    };

    const value: AuthContextType = {
        user,
        isLoading,
        isLoggedIn: !!user,
        login,
        register,
        logout,
        refreshUser,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}

/**
 * Hook to require authentication
 * Redirects to login page if not authenticated
 */
export function useRequireAuth(redirectTo: string = '/login') {
    const auth = useAuth();

    useEffect(() => {
        if (!auth.isLoading && !auth.isLoggedIn) {
            // Get current location for return redirect after login
            const currentPath = window.location.pathname;
            const returnTo = currentPath !== redirectTo ? `?returnTo=${encodeURIComponent(currentPath)}` : '';
            
            // Redirect to login page
            window.location.href = `${redirectTo}${returnTo}`;
        }
    }, [auth.isLoading, auth.isLoggedIn, redirectTo]);

    return auth;
}
